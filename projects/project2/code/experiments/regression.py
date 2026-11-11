# regression.py
"""
Part b: Regression on the noisy Runge function with a feed-forward neural
network, compared with the OLS benchmark from project 1.

Main functions
--------------
run()   Train everything, save results to regression.npz and make the figures
        by calling plot().
plot()  Load regression.npz and make the figures, so figures can be changed
        without retraining.

Pipeline
--------
1. Data   load_data()
   Runge data with noise, split in two steps:
     - get_runge_data() splits off the test set
     - the validation set is split off the rest (validation_size is a
       fraction of the non-test part)
   Train: fit the weights. Val: choose the learning rate.
   Test: used once per model, for the final test MSE.

2. OLS benchmark   ols_benchmark()
   Closed-form OLS with the best degree from project 1, fitted on train and
   evaluated on test. The validation set is not used, since the degree is
   already fixed and there is nothing to tune. OLS then fits on the same
   data as the networks.

3. Learning-rate analysis   run_learning_rate_analysis(), once per optimizer
   For every architecture:
     - train one network per learning rate, in parallel
       (train_and_validate() -> train_network(), sigmoid hidden layers)
     - keep the learning rate with the lowest validation MSE
       (diverged runs give nan and are skipped)
     - evaluate the chosen network on the test set

4. Save and plot   save_results(), print_summary(), plot()

Saved results (output/regression/results/regression.npz)
--------------------------------------------------------
Settings:  optimizers, arch_labels, learning_rates, best_degree, sigma
Test set:  x_test, y_test
OLS:       ols_test_pred, ols_test_mse
Per optimizer <opt>, with A = n_architectures, L = n_learning_rates:
    val_mse_<opt>      (A, L)        validation MSE of every run
    best_gamma_<opt>   (A,)          chosen learning rate
    test_mse_<opt>     (A,)          test MSE of the chosen run
    history_<opt>      (A, epochs)   training loss of the chosen run
    test_pred_<opt>    (A, n_test)   test predictions of the chosen run
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

from data import design_matrix, split_data, runge_function
from experiments.common import make_output_dirs, save_results
from activations import Sigmoid
from metrics import mse
from linear_models import ols_fit
from plotting_utils import save_figure, legend_below, scientific_label, HEATMAP_CMAP, mark_zeros
from matplotlib.ticker import LogFormatterMathtext
from matplotlib.colors import LogNorm
from .common import (
    get_runge_data,
    load_results,
    make_output_dirs,
    predict,
    save_results,
    train_network,
)

TASK = "regression"
RESULTS_NAME = "regression.npz"
OPTIMIZER_LABELS = {
    "gd": "GD",
    "sgd": "SGD",
    "momentum": "Momentum",
    "adagrad": "AdaGrad",
    "rmsprop": "RMSprop",
    "adam": "Adam",
}  # for plot titles and legends


# ------------------
# Helpers
# ------------------


def hidden_label(layers) -> str:
    """
    Makes label for hidden layers in the network so we can save results in a json file.
    i.e. [1,50,50,1] -> "[50,50]"
    """
    label = "[" + ",".join(str(l) for l in layers[1:-1]) + "]"
    return label


def load_data(cfg, n_points=None) -> dict:
    """
    Generate runge data and split into training / validation / test

    get_runge_data does the train/test split with test_size and seed from
    [data]. The validation set is then split off the training part, so
    validation_size is a fraction of the training part.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary specifying the data source and parameters.
    n_points : int, optional
        Number of data points to generate. If None, uses the value from cfg.

    Returns
    -------
    dict:
        "train", "val, "test" : (X, y) tuples
    """
    data = cfg["data"]
    if n_points is None:
        n_points = data["n_points"]
    x_fit, x_test, y_fit, y_test = get_runge_data(cfg, n_points=n_points)

    validation_size = data["validation_size"]
    random_state = data["seed"]

    x_train, x_val, y_train, y_val = split_data(
        x_fit, y_fit, test_size=validation_size, random_state=random_state
    )

    return {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "test": (x_test, y_test),
    }


# ------------------
# OLS Benchmark
# ------------------


def ols_benchmark(cfg, data) -> tuple:
    """
    OLS with best poly degree from p1.

    Fitted on train, tested on test set.
    Model scales X, y with training statistics and predict() returns
    values on original scale, as in p1.

    Returns
    -------
    dict:
        "model" : fitted model
        "test_mse" : MSE on test set
        "grid_pred" : predictions on grid for plotting
    """
    degree = cfg["regression"]["best_degree"]
    x_train, y_train = data["train"]
    x_test, y_test = data["test"]

    model = ols_fit(design_matrix(x_train, degree), y_train)
    test_pred = model.predict(design_matrix(x_test, degree))

    # grid for smooth plotting
    x_grid = np.linspace(-1, 1, 500)
    grid_pred = model.predict(design_matrix(x_grid, degree))

    return test_pred, mse(y_test, test_pred), grid_pred


# ----------------------
# Learning rate analysis
# ----------------------


def train_and_validate(cfg, layers, optimizer_name, gamma, data) -> tuple:
    """
    Train a network and compute its validation MSE. Used in parallel.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing the training settings.
    layers : list[int]
        Network architecture, including input and output layers.
    optimizer_name : str
        Name of the optimizer.
    gamma : float
        Learning rate.
    data : dict
        Dictionary containing training, validation, and test datasets.

    Returns
    -------
    net : NeuralNetwork
        Trained neural network.
    scaler : Scaler
        Scaler fitted to the training data.
    hist : array-like
        Training loss recorded at each epoch.
    val_mse : float
        Validation MSE calculated on the original target scale.
    """
    x_train, y_train = data["train"]
    x_val, y_val = data["val"]
    net, scaler, hist = train_network(
        cfg, layers, optimizer_name, gamma, x_train, y_train, hidden_activation=Sigmoid
    )
    return net, scaler, hist, mse(y_val, predict(net, scaler, x_val))


def run_learning_rate_analysis(cfg, data, optimizer_name) -> dict:
    """
    Evaluate different learning rates for each network architecture.
    For each architecture, train a network with every learning rate
    and select the model with the lowest validation MSE. Evaluate
    the selected model on the test set.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing the regression,
        training, and parallelization settings.
    data : dict
        Dictionary containing training, validation, and test datasets.
    optimizer_name : str
        Name of the optimizer used for training.

    Returns
    -------
    dict:
        - val_mse_{optimizer}: Validation MSE for each architecture
          and learning rate, with shape (n_arch, n_lr).
        - best_gamma_{optimizer}: Selected learning rate for each
          architecture, with shape (n_arch,).
        - test_mse_{optimizer}: Test MSE of the selected models,
          with shape (n_arch,).
        - history_{optimizer}: Training loss history of the selected
          models, with shape (n_arch, epochs).
        - test_pred_{optimizer}: Test predictions of the selected
          models, with shape (n_arch, n_test).
    """
    reg = cfg["regression"]
    n_jobs = cfg["parallel"]["n_jobs"]
    architectures = reg["architectures"]
    gammas = reg["learning_rates"]

    n_arch, n_lr = len(architectures), len(gammas)

    x_test, y_test = data["test"]

    # grid for smooth plotting
    x_grid = np.linspace(-1, 1, 500)
    grid_pred = np.full((n_arch, len(x_grid)), np.nan)

    # initialize with nan so that diverged runs are skipped
    val_mse = np.full((n_arch, n_lr), np.nan)
    best_gamma = np.full(n_arch, np.nan)
    test_mse = np.full(n_arch, np.nan)
    history = np.full((n_arch, cfg["training"]["epochs"]), np.nan)
    test_pred = np.full((n_arch, len(y_test)), np.nan)

    for i, layers in enumerate(architectures):
        best_val, best_model = np.inf, None

        # train learning rate in parallel
        runs = Parallel(n_jobs=n_jobs)(
            delayed(train_and_validate)(cfg, layers, optimizer_name, gamma, data)
            for gamma in gammas
        )

        for j, (gamma, (net, scaler, hist, val)) in enumerate(zip(gammas, runs)):
            val_mse[i, j] = val
            print(
                f"{optimizer_name:>8} {hidden_label(layers):>10} "
                f"gamma = {gamma:.0e}: validation MSE = {val_mse[i, j]:.4g}"
            )

            # select model with lowest validation MSE
            if val_mse[i, j] < best_val:  # False for nan, so diverged runs are skipped
                best_val, best_model = val_mse[i, j], (net, scaler)
                best_gamma[i] = gamma
                history[i, : len(hist)] = hist

        # evaluate best model on test data
        if best_model is not None:
            test_pred[i] = predict(*best_model, x_test)
            test_mse[i] = mse(y_test, test_pred[i])
            grid_pred[i] = predict(*best_model, x_grid)

    return {
        f"val_mse_{optimizer_name}": val_mse,
        f"best_gamma_{optimizer_name}": best_gamma,
        f"test_mse_{optimizer_name}": test_mse,
        f"history_{optimizer_name}": history,
        f"test_pred_{optimizer_name}": test_pred,
        f"grid_pred_{optimizer_name}": grid_pred,
    }


def print_summary(results) -> None:
    """
    Print the test MSE of OLS and of the chosen network for every optimizer
    and architecture.

    Completely made with Claude Opus 5.5

    Parameters
    ----------
    results : dict
        Results from run(), or loaded with load_results.
    """
    print("\n" + "=" * 86)
    print("TASK B: REGRESSION AND ARCHITECTURE STUDY")
    print("=" * 86)
    print(f"\nTest MSE (noise level sigma^2 = {float(results['sigma']) ** 2:.4g})")
    print(f"OLS, degree {int(results['best_degree'])}: {float(results['ols_test_mse']):.4g}")

    for opt in results["optimizers"]:
        for i, arch in enumerate(results["arch_labels"]):
            print(
                f"{opt:>8} {arch:>10} gamma = {results[f'best_gamma_{opt}'][i]:.0e}: "
                f"{results[f'test_mse_{opt}'][i]:.4g}"
            )


# -------------------------------------------------------------
# Plotting
# Docstrings and names were given to claude and then he wrote the plotting code.
# We just made some small changes to the code.
# -------------------------------------------------------------


def plot_lr_heatmap(val_mse, arch_labels, gammas, optimizer_name, fig_dir) -> None:
    """
    Heatmap of vaidation MSE for every architecture and learning rate.

    Parameters
    ----------
    val_mse : np.ndarray
        Validation MSE, shape (n_architectures, n_learning_rates).
        Diverged runs (nan) are left blank.
    arch_labels : list[str]
        Hidden layers of each architecture.
    gammas : list[float]
        Learning rates.
    optimizer_name : str
        Used in the title and file name.
    """
    fig, ax = plt.subplots(figsize=(7.06, 3.0))
    # log color scale: the MSE spans several orders of magnitude when some runs diverge
    sns.heatmap(
        val_mse,
        annot=True,
        fmt=".4f",
        annot_kws={"fontsize": 8},
        cmap=HEATMAP_CMAP,
        norm=LogNorm(vmin=np.nanmin(val_mse), vmax=np.nanmax(val_mse)),
        xticklabels=[scientific_label(gamma) for gamma in gammas],
        yticklabels=arch_labels,
        cbar_kws={"label": "Validation MSE"},
        ax=ax,
    )
    mark_zeros(ax, val_mse)
    ax.set_xlabel(r"Learning rate $\gamma$")
    ax.set_ylabel("Hidden layers")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    save_figure(fig, f"regression_lr_heatmap_{optimizer_name}.pdf", fig_dir=fig_dir)


def plot_loss_curves(histories, best_gammas, arch_labels, fig_dir) -> None:
    """
    Training loss per epoch for the chosen learning rate of each optimizer.
    One panel per optimizer, one curve per architecture. Maybe change to a single plot, but we see.

    Parameters
    ----------
    histories : dict
        {optimizer: array of shape (n_architectures, epochs)}.
    best_gammas : dict
        {optimizer: chosen learning rate for each architecture}.
    arch_labels : list[str]
        Hidden layers of each architecture.
    """
    fig, axes = plt.subplots(
        1, len(histories), figsize=(7.06, 3.0), sharey=True
    )
    axes = np.atleast_1d(axes)
    height = fig.get_figheight()

    for ax, (optimizer_name, history) in zip(axes, histories.items()):
        for i, arch in enumerate(arch_labels):
            gamma = best_gammas[optimizer_name][i]
            ax.plot(history[i], label=rf"{arch}, $\gamma$ = {scientific_label(gamma)}")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_title(OPTIMIZER_LABELS[str(optimizer_name)])
        legend_below(ax, ncol=1)
        ax.grid()
    # legend_below makes the figure taller once per panel, but the legends sit side by side
    fig.set_figheight(height + (fig.get_figheight() - height) / len(axes))
    axes[0].set_ylabel("Training MSE (scaled)")
    save_figure(fig, "regression_loss_curves.pdf", fig_dir=fig_dir)


def plot_regression_fit(
    x_grid,
    x_test,
    y_test,
    ols_grid_pred,
    nn_grid_pred,
    ols_label,
    nn_label,
    fig_dir,
) -> None:
    """
    Plot the Runge function and the fitted OLS and neural-network
    predictions on an evenly spaced grid.
    """
    fig, ax = plt.subplots()

    ax.scatter(
        x_test, y_test,
        s=10, color="gray", alpha=0.6,
        label="Test data",
    )
    ax.plot(
        x_grid,
        runge_function(x_grid),
        linestyle="--",
        color="black",
        label="Runge function",
    )
    ax.plot(x_grid, ols_grid_pred, label=ols_label)
    ax.plot(x_grid, nn_grid_pred, label=nn_label)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.grid()

    legend_below(ax, ncol=2)
    save_figure(fig, "regression_fit.pdf", fig_dir=fig_dir)

# ---------------------------------------------------------------------
# Run & Plot
# Also made with Claude Opus 5.5 based on docstring giving step by step.
# This is as it is just tedious work implementing the steps.
# ---------------------------------------------------------------------


def run(cfg: dict) -> None:
    """
    Run the complete regression analysis for part b, save the results and plot them.

    1. OLS benchmark with the best degree from project 1.
    2. Learning-rate analysis for every optimizer and architecture.
    3. Save everything to results/regression.npz and make the figures.

    Parameters
    ----------
    cfg : dict      Configuration dictionary.
    """
    _, result_dir = make_output_dirs(TASK)
    reg = cfg["regression"]

    data = load_data(cfg, n_points=reg["n_points"])
    x_test, y_test = data["test"]

    # 1. OLS benchmark
    ols_test_pred, ols_test_mse, ols_grid_pred = ols_benchmark(cfg, data)

    results = {
        # settings, so plot() does not need the config
        "optimizers": np.array(reg["optimizers"]),
        "arch_labels": np.array(
            [hidden_label(layers) for layers in reg["architectures"]]
        ),
        "learning_rates": np.array(reg["learning_rates"]),
        "best_degree": reg["best_degree"],
        "sigma": cfg["data"]["sigma"],
        # test data and OLS
        "x_test": x_test,
        "y_test": y_test,
        "ols_test_pred": ols_test_pred,
        "ols_test_mse": ols_test_mse,
        "x_grid": np.linspace(-1, 1, 500),
        "ols_grid_pred": ols_grid_pred,
    }

    # 2. Learning-rate analysis
    for optimizer_name in reg["optimizers"]:
        results.update(run_learning_rate_analysis(cfg, data, optimizer_name))

    # 3. Save, print and plot
    save_results(results, result_dir / RESULTS_NAME)
    print_summary(results)
    plot(cfg)


def plot_test_mse(test_mse, arch_labels, ols_test_mse, sigma, fig_dir) -> None:
    """
    Test MSE of the chosen network for every optimizer and architecture,
    compared with OLS and the noise level sigma^2.

    Parameters
    ----------
    test_mse : dict
        {optimizer: test MSE for each architecture}.
    arch_labels : list[str]
        Hidden layers of each architecture.
    ols_test_mse : float
        Test MSE of OLS with the best degree from project 1.
    sigma : float
        Standard deviation of the noise, sigma^2 is the lowest possible MSE.
    fig_dir : pathlib.Path
        Directory for the figure.
    """
    optimizers = list(test_mse)
    x = np.arange(len(arch_labels))
    width = 0.8 / len(optimizers)

    fig, ax = plt.subplots()
    for k, opt in enumerate(optimizers):
        offset = (k - (len(optimizers) - 1) / 2) * width  # bars side by side
        ax.bar(x + offset, test_mse[opt], width, label=OPTIMIZER_LABELS[str(opt)])
    ax.axhline(ols_test_mse, color="black", linestyle="--", label="OLS")
    ax.axhline(sigma**2, color="gray", linestyle=":", label=r"$\sigma^2$")
    ax.set_xticks(x, arch_labels)
    ax.set_xlabel("Hidden layers")
    ax.set_ylabel("Test MSE")
    ax.set_yscale("log")
    ax.grid()
    legend_below(ax, ncol=3)
    save_figure(fig, "regression_test_mse.pdf", fig_dir=fig_dir)


def plot(cfg=None) -> None:
    """
    Plot the results from the regression analysis.

    Only loads the results saved by run(), so the figures can be changed
    without training again.

    Parameters
    ----------
    cfg : dict      Configuration dictionary.
    """
    fig_dir, result_dir = make_output_dirs(TASK)
    results = load_results(result_dir / RESULTS_NAME)

    optimizers = results["optimizers"]
    arch_labels = results["arch_labels"]

    # Validation MSE, one heatmap per optimizer
    for opt in optimizers:
        plot_lr_heatmap(
            results[f"val_mse_{opt}"],
            arch_labels,
            results["learning_rates"],
            opt,
            fig_dir,
        )

    # Training loss of the chosen runs
    plot_loss_curves(
        {opt: results[f"history_{opt}"] for opt in optimizers},
        {opt: results[f"best_gamma_{opt}"] for opt in optimizers},
        arch_labels,
        fig_dir,
    )

    # Best network (lowest validation MSE of all runs) compared with OLS
    best_val, best_opt, best_i = np.inf, None, None
    for opt in optimizers:
        for i in range(len(arch_labels)):
            val = np.nanmin(results[f"val_mse_{opt}"][i])
            if val < best_val:
                best_val, best_opt, best_i = val, opt, i

    plot_regression_fit(
        results["x_grid"],
        results["x_test"],
        results["y_test"],
        results["ols_grid_pred"],
        results[f"grid_pred_{best_opt}"][best_i],
        ols_label=f"OLS, degree {int(results['best_degree'])}",
        nn_label=f"FFNN {arch_labels[best_i]}, {OPTIMIZER_LABELS[str(best_opt)]}",
        fig_dir=fig_dir,
    )

    # Test MSE of the chosen networks compared with OLS
    plot_test_mse(
        {opt: results[f"test_mse_{opt}"] for opt in optimizers},
        arch_labels,
        float(results["ols_test_mse"]),
        float(results["sigma"]),
        fig_dir,
    )
