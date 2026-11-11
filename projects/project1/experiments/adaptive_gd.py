# experiments/adaptive_gd.py

# ----------------------------------------------------------------------
# Task f: Momentum, AdaGrad, ADAM, and RMSProp for OLS and Ridge regression
# - Error relative to closed-form solution
# - Number of iterations needed
# - Sensitivity to learning rate
# -----------------------------------------------------------------------


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
from optimizers import (
    max_learning_rate,
)


def error_vs_closed_form(
    cfg: dict, method: str, optimizer_name: str, learning_rate: float
):
    """
    Run the specified optimizer for OLS or Ridge regression and compute the relative error
    with respect to the closed-form solution.

    Parameters
    ----------
    cfg : dict                  Configuration dictionary.
    method : str                Regression method ("ols" or "ridge").
    optimizer_name : str        Name of the optimizer ("GD", "momentum", "adagrad", "rmsprop", or "adam").
    learning_rate : float       Learning rate for the optimizer.

    Returns
    -------
    np.ndarray : Relative error in each iteration with respect to the closed-form solution.
    """

    adaptive_gd_cfg = cfg["adaptive_gd"]
    max_iter = adaptive_gd_cfg["max_iter"]
    tol = adaptive_gd_cfg["tol"]

    X, y, lmbda = get_gd_data(cfg, method)
    closed_form_model = Model.fit(X, y, method=method, lmbda=lmbda)

    errors = []

    def record_error(theta):
        errors.append(rel_error(theta, closed_form_model.theta))

    optimizer, optimizer_params = optimizer_settings(optimizer_name, learning_rate, cfg)
    with np.errstate(
        divide="ignore", invalid="ignore", over="ignore"
    ):  # ignore warnings from diverging runs (too large learning rate); they are marked "div."
        Model.fit(
            X,
            y,
            method=method,
            lmbda=lmbda,
            optimizer=optimizer,
            max_iter=max_iter,
            tol=tol,
            recorder_func=record_error,
            **optimizer_params,
        )
    return np.array(errors)


def run_learning_rate_sensitivity(cfg: dict, method: str):
    """
    Run every optimizer for every learning rate for given method.

    Returns
    -------
    learning_rates : np.ndarray     Array of learning rates tested.
    iterations : np.ndarray          2D array of iterations needed for each optimizer and learning rate.
    final_errors : np.ndarray        2D array of final errors for each optimizer and learning rate.
    """

    adaptive_gd_cfg = cfg["adaptive_gd"]
    learning_rates = np.asarray(adaptive_gd_cfg["learning_rates"])
    target_error = adaptive_gd_cfg["target_error"]

    shape = (len(OPTIMIZER_NAMES), len(learning_rates))
    iterations = np.zeros(shape)
    final_errors = np.zeros(shape)

    for i, name in enumerate(OPTIMIZER_NAMES):
        for j, learning_rate in enumerate(learning_rates):
            errors = error_vs_closed_form(cfg, method, name, learning_rate)
            iterations[i, j] = iterations_for_target(errors, target_error)
            final_error = errors[-1]
            final_errors[i, j] = final_error if np.isfinite(final_error) else np.nan

    return learning_rates, iterations, final_errors


def best_learning_rates(
    learning_rates: np.ndarray, iterations: np.ndarray, final_errors: np.ndarray
):
    """
    Determine the best learning rate for each optimizer: the one that reaches the
    target error in the fewest iterations. If an optimizer never reaches the target,
    the learning rate with the smallest final error is used instead. NaN if the
    optimizer diverged for every learning rate.

    Parameters
    ----------
    learning_rates : np.ndarray     Array of learning rates tested.
    iterations : np.ndarray          2D array of iterations needed for each optimizer and learning rate.
    final_errors : np.ndarray        2D array of final errors for each optimizer and learning

    Returns
    -------
    best : list                 Best learning rate for each optimizer, in the order of OPTIMIZER_NAMES.
    """
    best = []
    for i in range(len(OPTIMIZER_NAMES)):
        if np.any(
            np.isfinite(iterations[i])
        ):  # reached the target for some learning rate
            best.append(learning_rates[np.nanargmin(iterations[i])])
        elif np.all(np.isnan(final_errors[i])):  # diverged for every learning rate
            best.append(np.nan)
        else:
            best.append(learning_rates[np.nanargmin(final_errors[i])])
    return best


def iterations_table(learning_rates, iterations, final_errors, method, target):
    """
    Print the number of iterations needed to reach the target error for every
    optimizer and learning rate. "-" means the target was not reached, "div." that
    the run diverged.
    """

    def cell(i, j):
        if np.isnan(final_errors[i, j]):
            return "div."
        if np.isnan(iterations[i, j]):
            return "-"
        return f"{int(iterations[i, j])}"

    print(f"\n{method.upper()}: iterations to reach relative error {target:.0e}")
    print(f"{'Optimizer':<10}" + "".join(f"{gamma:>8g}" for gamma in learning_rates))
    for i, name in enumerate(OPTIMIZER_NAMES):
        print(
            f"{name:<10}"
            + "".join(f"{cell(i, j):>8}" for j in range(len(learning_rates)))
        )


def run_adaptive_gd(cfg: dict) -> None:
    """
    Run the complete adaptive gradient descent analysis for task f, for OLS and Ridge:

    1. Learning-rate sensitivity: iterations to reach the target error and final
       error for every optimizer and learning rate (printed table and two plots).
    2. Convergence curves (error vs iteration) for every optimizer at its best
       learning rate.
    """
    adaptive_gd_cfg = cfg["adaptive_gd"]
    target = adaptive_gd_cfg["target_error"]
    max_iter = adaptive_gd_cfg["max_iter"]
    iterations_by_method = {}
    gamma_max_by_method = {}

    for method in ("ols", "ridge"):
        # stability limit for the GD learning rate, shown in the plots
        X, y, lmbda = get_gd_data(cfg, method)
        closed_form_model = Model.fit(X, y, method=method, lmbda=lmbda)
        X_scaled = closed_form_model.scale_X(
            X, closed_form_model.mean_X, closed_form_model.std_X
        )
        gamma_max = max_learning_rate(X_scaled, method=method, lmbda=lmbda)[1]
        gamma_max_by_method[method] = gamma_max

        # 1. learning-rate sensitivity
        learning_rates, iterations, final_errors = run_learning_rate_sensitivity(
            cfg, method
        )
        best = best_learning_rates(learning_rates, iterations, final_errors)

        print(
            f"\n{method.upper()}: optimizers (target relative error {target:.0e}, "
            f"max {max_iter} iterations, gamma_max = {gamma_max:.3f})"
        )
        print(
            "-----------------------------------------------------------------------------"
        )
        print(
            f"{'Optimizer':<10} {'best lr':>9} {'iterations':>11} {'final error':>12}   "
            f"{'lr range reaching target':<24}"
        )

        for i, name in enumerate(OPTIMIZER_NAMES):
            if np.isnan(best[i]):  # diverged for every learning rate
                print(f"{name:<10} {'-':>9} {'-':>11} {'diverged':>12}")
                continue
            j = list(learning_rates).index(best[i])  # column of the best learning rate
            reached = learning_rates[np.isfinite(iterations[i])]
            if len(reached) > 0:
                lr_range = ", ".join(f"{lr:.0e}" for lr in reached)
            else:
                lr_range = "never"
            n_iter = "-" if np.isnan(iterations[i, j]) else f"{int(iterations[i, j])}"
            print(
                f"{name:<10} {best[i]:9.1e} {n_iter:>11} {final_errors[i, j]:12.2e}   {lr_range:<24}"
            )

        iterations_table(learning_rates, iterations, final_errors, method, target)
        iterations_by_method[method] = iterations

        plot.plot_learning_rate_sensitivity(
            learning_rates,
            iterations,
            OPTIMIZER_NAMES,
            ylabel=f"Iterations to reach relative error {target}",
            method=method,
            name="iterations_to_target",
            gamma_max=gamma_max,
        )
        plot.plot_learning_rate_sensitivity(
            learning_rates,
            final_errors,
            OPTIMIZER_NAMES,
            ylabel=f"Relative error after {max_iter} iterations",
            method=method,
            name="final_error",
            gamma_max=gamma_max,
        )

        # 2. convergence curves at each optimizer's best learning rate
        converged = [
            (name, lr) for name, lr in zip(OPTIMIZER_NAMES, best) if not np.isnan(lr)
        ]  # skip optimizers that diverged for every learning rate
        histories = [
            error_vs_closed_form(cfg, method, name, lr) for name, lr in converged
        ]
        labels = [rf"{name} ($\gamma$)" for name, _ in converged]
        plot.plot_convergence(
            histories, labels, method=method, target=target, smooth_window=100, ncol=2
        )

    plot.plot_learning_rate_sensitivity_ols_ridge(
        learning_rates,
        iterations_by_method,
        OPTIMIZER_NAMES,
        ylabel=f"Iterations to reach relative error {target}",
        name="iterations_to_target",
        gamma_max=gamma_max_by_method["ols"],  # gamma_max for OLS is same as for ridge
    )
