# experiments/stochastic_gd.py

# -------------------------------------------------------------------
# Task h: Stochastic gradient descent for OLS and Ridge
# 1. Accuracy after a fixed number of epochs, and wall time,
#    for different mini-batch sizes.
# 2. Number of epochs: error per epoch for SGD vs full batch
# 3. Learning rate schedule. Compare constant and inverse time decay.
# 4. Computational cost (epoch and wall time to reach a given accuracy) for different optimizers and learning rates.
# -------------------------------------------------------------------

import time
import numpy as np

from .common import (
    get_gd_data,
    OPTIMIZER_NAMES,
    optimizer_settings,
    iterations_for_target,
)
from metrics import rel_error
from models import Model
import plotting as plot

# ---------------------------------------------------------------
# helpers
# ---------------------------------------------------------------


def updates_per_epoch(n_samples, batch_size):
    """Compute the number of updates per epoch given the number of samples and batch size."""
    return int(np.ceil(n_samples / batch_size))


def fit_sgd(cfg, method, optimizer_name, batch_size, schedule=None, recorder_func=None):
    """
    Fit OLS or Ridge with given optimizer
    """
    sgd_cfg = cfg["sgd"]
    X, y, lmbda = get_gd_data(cfg, method)
    gamma = sgd_cfg["gammas"][optimizer_name]
    optimizer, optimizer_params = optimizer_settings(
        optimizer_name, gamma, cfg, batch_size=batch_size, schedule=schedule
    )

    start = time.perf_counter()  # start timer
    with np.errstate(
        over="ignore", invalid="ignore"
    ):  # ignore warnings from diverging runs (e.g. M = 1); they are marked "diverged"
        model = Model.fit(
            X,
            y,
            method=method,
            lmbda=lmbda,
            optimizer=optimizer,
            n_epochs=sgd_cfg["n_epochs"],
            tol=0.0,  # never stop early as every run uses the same number of epochs
            random_state=cfg["data"][
                "seed"
            ],  # seed for shuffling the data in each epoch
            recorder_func=recorder_func,
            **optimizer_params,
        )
    wall_time = time.perf_counter() - start  # stop timer
    return model, wall_time


def make_schedule_func(decay_updates):
    """
    Inverse time decay schedule: gamma_t = gamma_0 / (1 + t / decay_updates)

    Parameters
    ----------
    decay_updates : int
        Number of updates after which the learning rate is halved.

    Returns
    -------
    schedule : callable
        A function that takes the current update number t and returns the learning rate multiplier.
    """

    def schedule(t):
        return decay_updates / (decay_updates + t)

    return schedule


def error_per_epoch(cfg, method, optimizer_name, batch_size, schedule=None):
    """
    Compute the relative error per epoch for SGD vs full batch
    """
    sgd_cfg = cfg["sgd"]

    X, y, lmbda = get_gd_data(cfg, method)
    closed_form_model = Model.fit(X, y, method=method, lmbda=lmbda)

    errors = []

    def record_error(theta):
        errors.append(rel_error(theta, closed_form_model.theta))

    record_error.epoch = 0

    fit_sgd(
        cfg,
        method,
        optimizer_name,
        batch_size,
        schedule=schedule,
        recorder_func=record_error,
    )

    per_epoch = updates_per_epoch(len(y), batch_size)
    errors = np.array(errors)[
        per_epoch - 1 :: per_epoch
    ]  # take the last error of each epoch
    return errors


# ---------------------------------------------------------------
# 1. Accuracy after a fixed number of epochs, and wall time,
#    for different mini-batch sizes.
# ---------------------------------------------------------------


def run_batch_size_comparison(cfg, method):
    """
    Run every optimizer for every mini-batch size for n_epochs and record the relative error and wall time.
    """
    X, y, lmbda = get_gd_data(cfg, method=method)
    closed_form_model = Model.fit(X, y, method=method, lmbda=lmbda)

    batch_sizes = list(cfg["sgd"]["batch_sizes"]) + [
        None
    ]  # None is full batch (no SGD)
    shape = (len(OPTIMIZER_NAMES), len(batch_sizes))
    final_errors = np.zeros(shape)
    wall_times = np.zeros(shape)

    for i, name in enumerate(OPTIMIZER_NAMES):
        for j, batch_size in enumerate(batch_sizes):
            model, wall_time = fit_sgd(
                cfg, method=method, optimizer_name=name, batch_size=batch_size
            )
            error = rel_error(model.theta, closed_form_model.theta)
            final_errors[i, j] = error
            wall_times[i, j] = wall_time

    return batch_sizes, final_errors, wall_times


# ---------------------------------------------------------------
# 2. & 3. Number of epochs and learning rate schedule
# ---------------------------------------------------------------


def run_epoch_and_schedule_comparison(cfg: dict, method: str):
    """
    Run every optimizer for a fixed mini-batch size and record the relative error per epoch for different learning rate schedules.
    """

    sgd_cfg = cfg["sgd"]
    batch_size = sgd_cfg["batch_size"]
    X, _, _ = get_gd_data(
        cfg, method=method
    )  # only need X to compute the number of updates per epoch

    # decay as epochs
    decay_epochs = sgd_cfg["decay_epochs"]
    decay_updates = updates_per_epoch(len(X), batch_size) * decay_epochs
    schedule = make_schedule_func(decay_updates)

    sgd_const, sgd_decay, full_batch = [], [], []

    for name in OPTIMIZER_NAMES:
        sgd_const.append(error_per_epoch(cfg, method, name, batch_size, schedule=None))
        sgd_decay.append(
            error_per_epoch(cfg, method, name, batch_size, schedule=schedule)
        )
        full_batch.append(error_per_epoch(cfg, method, name, len(X), schedule=None))

    return np.array(sgd_const), np.array(sgd_decay), np.array(full_batch)


# ---------------------------------------------------------------
# run everything
# ---------------------------------------------------------------


def run_sgd(cfg: dict) -> None:
    """
    Run the stochastic gradient descent experiments for OLS and Ridge regression, comparing different mini-batch sizes and optimizers.
    Largely written with help of Claude Sonnet 5 to get the plotting and printing to work nicely.
    """
    sgd_cfg = cfg["sgd"]
    n_epochs = sgd_cfg["n_epochs"]
    batch_size = sgd_cfg["batch_size"]
    target = sgd_cfg["target_error"]

    for method in ("ols", "ridge"):
        X, _, _ = get_gd_data(cfg, method)
        n = len(X)

        # 1. mini-batch size
        batch_sizes, final_errors, wall_times = run_batch_size_comparison(
            cfg, method=method
        )
        sizes = [
            n if M is None else M for M in batch_sizes
        ]  # full batch shown as M = n

        print(
            f"\n{method.upper()}: SGD, relative error to closed form after {n_epochs} epochs"
        )
        print(
            "-----------------------------------------------------------------------------"
        )
        header = " ".join(
            f"{('M=' + str(M)) if M is not None else 'full':>10}" for M in batch_sizes
        )
        print(f"{'Optimizer':<10} {header}")
        for i, name in enumerate(OPTIMIZER_NAMES):
            row = " ".join(
                f"{'diverged':>10}" if np.isnan(e) else f"{e:10.2e}"
                for e in final_errors[i]
            )
            print(f"{name:<10} {row}")
        times = " ".join(f"{t:10.2f}" for t in wall_times.mean(axis=0))
        print(f"{'time [s]':<10} {times}   (mean over optimizers)")

        plot.plot_error_vs_batch_size(
            sizes, final_errors, OPTIMIZER_NAMES, n_epochs, method=method
        )

        # 2. and 3. epochs and learning-rate schedule
        sgd_constant, sgd_decay, full_batch = run_epoch_and_schedule_comparison(
            cfg, method
        )

        plot.plot_error_vs_epoch(
            sgd_constant,
            full_batch,
            OPTIMIZER_NAMES,
            solid_label=f"SGD, M = {batch_size}",
            dashed_label="Full batch",
            method=method,
            name=f"sgd_vs_full_M{batch_size}",
            target=target,
            smooth_window=50,
        )
        plot.plot_error_vs_epoch(
            sgd_constant,
            sgd_decay,
            OPTIMIZER_NAMES,
            solid_label="Constant learning rate",
            dashed_label=f"Inverse time decay ({sgd_cfg['decay_epochs']} epochs)",
            method=method,
            name=f"schedule_M{batch_size}",
            target=target,
            smooth_window=50,
        )

        # 4. accuracy and computational cost, SGD vs full batch
        j_sgd = batch_sizes.index(batch_size)
        j_full = batch_sizes.index(None)

        print(
            f"\n{method.upper()}: SGD (M = {batch_size}) vs full batch, {n_epochs} epochs, "
            f"target relative error {target:.0e}"
        )
        print("-" * 108)
        print(
            f"{'Optimizer':<10} {'final error':>34} {'epochs to target':>20} "
            f"{'time/epoch [ms]':>18} {'time to target [ms]':>22}"
        )
        print(
            f"{'':<10} {'full':>10} {'SGD':>10} {'SGD+decay':>12} {'full':>9} {'SGD':>9} "
            f"{'full':>9} {'SGD':>8} {'full':>11} {'SGD':>10}"
        )

        def fmt_err(e):
            return f"{'diverged':>10}" if np.isnan(e) else f"{e:10.2e}"

        def fmt_ep(e):
            return f"{'-':>9}" if np.isnan(e) else f"{int(e):9d}"

        def fmt_ms(t, width=10):
            return f"{'-':>{width}}" if np.isnan(t) else f"{t:{width}.1f}"

        for i, name in enumerate(OPTIMIZER_NAMES):
            errors_full = full_batch[i][-1]
            errors_sgd = sgd_constant[i][-1]
            errors_decay = sgd_decay[i][-1]
            epochs_full = iterations_for_target(full_batch[i], target)
            epochs_sgd = iterations_for_target(sgd_constant[i], target)
            ms_full = 1000 * wall_times[i, j_full] / n_epochs
            ms_sgd = 1000 * wall_times[i, j_sgd] / n_epochs

            # total time to reach the target = epochs to target * time per epoch
            # (NaN if the target was never reached, printed as "-")
            total_full = epochs_full * ms_full
            total_sgd = epochs_sgd * ms_sgd

            print(
                f"{name:<10} {fmt_err(errors_full)} {fmt_err(errors_sgd)} {fmt_err(errors_decay):>12} "
                f"{fmt_ep(epochs_full)} {fmt_ep(epochs_sgd)} {ms_full:9.3f} {ms_sgd:8.3f} "
                f"{fmt_ms(total_full, 11)} {fmt_ms(total_sgd, 10)}"
            )
