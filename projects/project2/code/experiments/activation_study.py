# activation_study.py
"""
Task c: study affect of activation function and architecture (depth and width).

We compare Sigmoid, ReLU and LeakyReLU by computing the MSE for training and
validation data as functions of the number of hidden layers (depth) and
nodes (width).

For each activation function, we select the architecture giving the lowest validation MSE
and compute the test MSE.

run() trains the networks, saves results and plots them.
only loads saved results and makes figures.



Main functions
--------------
run()   Train everything, save results to activation_study.npz, print a
        summary and make the figures by calling plot().
plot()  Load activation_study.npz and make the figures, so figures can be
        changed without retraining.

Pipeline
---------------
1. Data load_data() from experiments.regression
    - Same train/val/test split as in task b (experiments/regression.py)
    - train: fit the weights
    - validation: select the best architecture for each activation function
    - test: evaluate the selected architecture for each activation function

2. Activation and architecture study run_activation_architecture_study()
    For each activation function:
        - for every depth and width, train one network 
          with layers = [1] + depth * [width] + [1], in parallel
        - record train, and val MSE and training loss history
        - keep architecture with lowest validation MSE
        - evaluate the selected architecture on the test set and record test MSE

Saved results (output/activation_study/results/activation_study.npz)
--------------------------------------------------------------------
Settings:  activations, depths, widths, optimizer, learning_rate, epochs
Test data: x_test, y_test, x_grid
Per run, with N = n_activations, D = n_depths, W = n_widths:
    train_mse    (N, D, W)           training MSE
    val_mse      (N, D, W)           validation MSE
    histories    (N, D, W, epochs)   training loss per epoch
Per activation, for the chosen architecture:
    test_mse     (N,)                test MSE
    test_pred    (N, n_test)         test predictions
    grid_pred    (N, n_grid)         predictions on x_grid
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

from activations import Sigmoid, ReLU, LeakyReLU
from data import runge_function
from experiments.regression import load_data
from experiments.classification import ACTIVATION_LABELS
from metrics import mse
from plotting_utils import save_figure, legend_below, HEATMAP_CMAP, mark_zeros, highlight_cell
from matplotlib.lines import Line2D
from experiments.common import (
    train_network,
    predict,
    make_output_dirs,
    save_results,
    load_results,
)

TASK = "activation_study"
RESULTS_NAME = "activation_study.npz"

ACTIVATIONS = {
    "sigmoid": Sigmoid,
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
}


# ------------------------------------------
# Activation function and architecture study
# ------------------------------------------


def train_and_evaluate(cfg, layers, activation_name, data) -> tuple:
    """
    Train a neural network with chosen activation function
    and evaluate the training and validation MSE.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.
    layers : list[int]
        Network architecture, including input and output layers.
    activation_name : str
        Hidden-layer activation function.
    data : dict
        Training, validation and test datasets.

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
    study = cfg["activation_study"]

    x_train, y_train = data["train"]
    x_val, y_val = data["val"]

    net, scaler, history = train_network(
        cfg,
        layers,
        optimizer_name=study["optimizer"],
        gamma=study["learning_rate"],
        x_train=x_train,
        y_train=y_train,
        hidden_activation=ACTIVATIONS[activation_name],
    )
    train_mse = mse(y_train, predict(net, scaler, x_train))
    val_mse = mse(y_val, predict(net, scaler, x_val))

    return net, scaler, train_mse, val_mse, history


def run_activation_architecture_study(cfg) -> dict:
    """
    Compare activation functions across different architectures (depths and widths).
    For each of the activation functions, we choose the architecture with the
    lowest validation MSE and evaluate it on the test set (compute test MSE).

    Returns
    -------
    dict
        Training MSE, validation MSE, test MSE, training histories and selected architecture.
    """
    study = cfg["activation_study"]
    n_jobs = cfg["parallel"]["n_jobs"]

    data = load_data(cfg, n_points=cfg["activation_study"]["n_points"])
    x_test, y_test = data["test"]

    activations = study["activations"]
    depths = study["depths"]  # hidden layers
    widths = study["widths"]  # number of nodes in each hidden layer
    epochs = cfg["training"]["epochs"]

    # holding grid values for plotting the fit
    x_grid = np.linspace(-1, 1, 500)
    grid_pred = np.full((len(activations), len(x_grid)), np.nan)

    # arrays for holding values using nan values, not zeros that can affect MSE interpretation
    train_mse = np.full((len(activations), len(depths), len(widths)), np.nan)
    val_mse = np.full_like(train_mse, np.nan)
    history = np.full((len(activations), epochs), np.nan)
    best_depth = np.full(len(activations), -1, dtype=int)
    best_width = np.full(len(activations), -1, dtype=int)

    # store test mse for the selected architecture for each activation
    test_mse = np.full(len(activations), np.nan)
    test_pred = np.full((len(activations), len(y_test)), np.nan)

    for a, activation_name in enumerate(activations):
        best_val, best_model = np.inf, None

        # construct all architectures for parallelization
        architectures = [
            (d, w, depth, width)
            for d, depth in enumerate(depths)
            for w, width in enumerate(widths)
        ]

        # train all architectures in parallel
        runs = Parallel(n_jobs=n_jobs)(
            delayed(train_and_evaluate)(
                cfg,
                layers=[1] + depth * [width] + [1],
                activation_name=activation_name,
                data=data)
            for d, w, depth, width in architectures
        )

        # recover results correspoding to each architecture
        for (d, w, depth, width), result in zip(architectures, runs):
            net, scaler, train_mse[a, d, w], val_mse[a, d, w], hist = result
            print(
                f"{activation_name:>12} "
                f"depth={depth}, width={width}: "
                f"train={train_mse[a,d,w]:.5f}, val={val_mse[a,d,w]:.5f}"
            )
            # select architecture with lowest validation MSE
            if np.isfinite(val_mse[a, d, w]) and val_mse[a, d, w] < best_val:  # avoids NaN issues
                best_val, best_model = val_mse[a, d, w], (net, scaler)
                best_depth[a], best_width[a] = depth, width
                history[a, :] = np.nan # to ensure each row is full
                history[a, :len(hist)] = hist

        # evaluate best model on test data
        if best_model is not None:
            test_pred[a] = predict(*best_model, x_test)
            test_mse[a] = mse(y_test, test_pred[a])
            grid_pred[a] = predict(*best_model, x_grid)

    return {
        "train_mse": train_mse,
        "val_mse": val_mse,
        "histories": history,
        "best_depth": best_depth,
        "best_width": best_width,
        "test_mse": test_mse,
        "test_pred": test_pred,
        "x_test": x_test,
        "y_test": y_test,
        "x_grid": x_grid,
        "grid_pred": grid_pred,
    }


# this function was edited by ChatGPT-6 for better formatting
def print_summary(results) -> None:
    """
    Print the best architecture for each activation function,
    selected using validation MSE, with train, validation and test MSE.
    """
    print("\n" + "=" * 86)
    print("TASK C: ACTIVATION FUNCTION AND ARCHITECTURE STUDY")
    print("=" * 86)
    print(
        f"{'Activation':<15}"
        f"{'Depth':>8}"
        f"{'Width':>8}"
        f"{'Train MSE':>15}"
        f"{'Val MSE':>15}"
        f"{'Test MSE':>15}"
    )
    print("-" * 86)

    for a, activation_name in enumerate(results["activations"]):
        val_mse = results["val_mse"][a]

        if not np.isfinite(val_mse).any():
            continue

        d, w = np.unravel_index(np.nanargmin(val_mse), val_mse.shape)
        print(
            f"{activation_name:<15}"
            f"{int(results['depths'][d]):>8}"
            f"{int(results['widths'][w]):>8}"
            f"{results['train_mse'][a, d, w]:>15.5f}"
            f"{val_mse[d, w]:>15.5f}"
            f"{results['test_mse'][a]:>15.5f}"
        )
    print("=" * 86)


# -------------------------------------------------------------
# Plotting:
# plotting functions made by ChatGPT-6 based on the
# docstrings and detailed prompts.
# -------------------------------------------------------------

# for en og en heatmap, vet ikke hva vi vil ha i rapporten:
# def plot_architecture_heatmaps(results, fig_dir) -> None:
#     """
#     Plot validation MSE as a function of depth and width.
#     """
#     val_mse = results["val_mse"]

#     vmin = np.nanmin(val_mse)
#     vmax = np.nanmax(val_mse)

#     for a, activation_name in enumerate(results["activations"]):
#         values = val_mse[a]

#         fig, ax = plt.subplots(figsize=(6, 4))

#         sns.heatmap(
#             values,
#             annot=True,
#             fmt=".4f",
#             vmin=vmin,
#             vmax=vmax,
#             xticklabels=results["widths"],
#             yticklabels=results["depths"],
#             cbar_kws={"label": "Validation MSE"},
#             ax=ax,
#         )
#         ax.set_xlabel("Neurons per hidden layer")
#         ax.set_ylabel("Number of hidden layers")
#         ax.set_title(activation_name)
#         save_figure(fig, f"activation_heatmap_{activation_name}.pdf", fig_dir=fig_dir)


# plot all heatmaps in subfigure
def plot_architecture_heatmaps(results, fig_dir) -> None:
    """
    Plot validation MSE for different depths and widths.
    Highlight the best architecture for each activation function.
    """
    activations = results["activations"]
    val_mse = results["val_mse"]

    vmin = np.nanmin(val_mse)
    vmax = np.nanmax(val_mse)

    fig = plt.figure(figsize=(7.06, 2.8))

    gs = fig.add_gridspec(
        1,
        len(activations) + 1,
        width_ratios=[1] * len(activations) + [0.05],
        wspace=0.05,
    )

    axes = [fig.add_subplot(gs[0, i]) for i in range(len(activations))]
    cbar_ax = fig.add_subplot(gs[0, -1])

    for a, (ax, activation_name) in enumerate(zip(axes, activations)):
        sns.heatmap(
            val_mse[a],
            annot=True,
            fmt=".5f",
            cmap=HEATMAP_CMAP,
            vmin=vmin,
            vmax=vmax,
            xticklabels=results["widths"],
            yticklabels=results["depths"],
            cbar=(a == len(activations) - 1),
            cbar_ax=cbar_ax if a == len(activations) - 1 else None,
            cbar_kws={"label": "Validation MSE"},
            ax=ax,
        )
        mark_zeros(ax, val_mse[a])

        # Highlight the architecture with lowest validation MSE
        if np.isfinite(val_mse[a]).any():
            best_row, best_col = np.unravel_index(
                np.nanargmin(val_mse[a]), val_mse[a].shape
            )
            highlight_cell(ax, best_col, best_row)
        ax.set_title(ACTIVATION_LABELS[str(activation_name)])
        ax.set_xlabel("Neurons per hidden layer")
        ax.tick_params(axis="y", rotation=0)
        if a > 0:  # Only show y-axis ticks and labels on the first heatmap
            ax.tick_params(axis="y", left=False, labelleft=False)

    axes[0].set_ylabel("Number of hidden layers")
    save_figure(fig, "activation_heatmaps.pdf", fig_dir=fig_dir)


def plot_depth_study(results, fig_dir) -> None:
    """
    Plot validation MSE vs network depth for fixed widths.
    One panel is generated for each activation function.
    """
    activations = results["activations"]
    depths = results["depths"]
    widths = results["widths"]

    fig, axes = plt.subplots(
        1,
        len(activations),
        figsize=(7.06, 2.8),
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    for a, (ax, activation_name) in enumerate(zip(axes, activations)):
        for w, width in enumerate(widths):
            ax.plot(
                depths,
                results["val_mse"][a, :, w],
                marker="o",
                label=f"{width} nodes",
            )
        ax.set_xlabel("Number of hidden layers")
        ax.set_title(ACTIVATION_LABELS[str(activation_name)])
        ax.set_xticks(depths)
        ax.grid()
        axes[0].set_ylabel("Validation MSE")

        handles, labels = axes[0].get_legend_handles_labels()

        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=len(widths),
            frameon=True,
        )
        fig.subplots_adjust(bottom=0.19)
        save_figure(fig, "activation_depth_study.pdf", fig_dir=fig_dir, Tight=False)


def plot_width_study(results, fig_dir) -> None:
    """
    Plot validation MSE versus network width for fixed depths.
    One panel is generated for each activation function.
    """
    activations = results["activations"]
    depths = results["depths"]
    widths = results["widths"]

    fig, axes = plt.subplots(
        1,
        len(activations),
        figsize=(7.06, 3.0),
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    for a, (ax, activation_name) in enumerate(zip(axes, activations)):
        for d, depth in enumerate(depths):
            ax.plot(
                widths,
                results["val_mse"][a, d, :],
                marker="o",
                label=f"{depth} layers",
            )
        ax.set_xlabel("Neurons per hidden layer")
        ax.set_title(ACTIVATION_LABELS[str(activation_name)])
        ax.set_xticks(widths)
        ax.grid()

    axes[0].set_ylabel("Validation MSE")

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(depths),
        frameon=True,
    )
    fig.subplots_adjust(bottom=0.19)
    save_figure(fig, "activation_width_study.pdf", fig_dir=fig_dir, Tight=False)


def plot_overfitting(results, fig_dir) -> None:
    """
    Compare training and validation MSE across network depths.

    Colors indicate network width, while line styles distinguish
    training and validation MSE.
    """
    activations = results["activations"]
    depths = results["depths"]
    widths = results["widths"]

    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(
        1,
        len(activations),
        figsize=(7.06, 3.0),
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    for a, (ax, activation_name) in enumerate(zip(axes, activations)):
        for w, width in enumerate(widths):
            color = colors[w % len(colors)]

            ax.plot(
                depths,
                results["train_mse"][a, :, w],
                "--",
                marker="o",
                color=color,
            )
            ax.plot(
                depths,
                results["val_mse"][a, :, w],
                "-",
                marker="o",
                color=color,
                label=f"{width} nodes",
            )
        ax.set_xlabel("Number of hidden layers")
        ax.set_title(ACTIVATION_LABELS[str(activation_name)])
        ax.set_xticks(depths)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("MSE")

    handles = [
        Line2D([0], [0], color=colors[w], lw=1.5, label=f"{width} nodes")
        for w, width in enumerate(widths)
    ]
    handles += [
        Line2D([0], [0], color="black", linestyle="--", lw=1.5, label="Training"),
        Line2D([0], [0], color="black", linestyle="-", lw=1.5, label="Validation"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=5,
        frameon=True,
    )
    fig.subplots_adjust(bottom=0.19)
    save_figure(fig, "activation_overfitting.pdf", fig_dir=fig_dir, Tight=False)


def plot_loss_curves(results, fig_dir) -> None:
    """
    Plot training loss per epoch for the best-validation
    architecture of each activation function.

    Loss is calculated on scaled targets.
    """
    fig, ax = plt.subplots()

    for a, activation_name in enumerate(results["activations"]):
        values = results["val_mse"][a]

        if not np.isfinite(values).any():
            continue

        d, w = np.unravel_index(np.nanargmin(values), values.shape)

        history = results["histories"][a]

        ax.plot(history,
            label=(
                f"{ACTIVATION_LABELS[str(activation_name)]}, "
                f"depth={results['depths'][d]}, "
                f"width={results['widths'][w]}"
            ),
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training MSE (scaled)")
    ax.set_yscale("log")
    ax.grid()
    legend_below(ax, ncol=2)
    save_figure(fig, "activation_loss_curves.pdf", fig_dir=fig_dir)


def plot_regression_fit(results, fig_dir) -> None:
    """
    Compare the true Runge function with the predictions of the
    validation-selected networks for each activation function.
    """
    x_grid = results["x_grid"]
    x_test = results["x_test"]
    y_test = results["y_test"]

    fig, ax = plt.subplots()

    # Noisy test observations
    ax.scatter(
        x_test,
        y_test,
        s=12,
        color="gray",
        alpha=0.5,
        label="Test data",
        zorder=1,
    )

    # True Runge function
    ax.plot(
        x_grid,
        runge_function(x_grid),
        color="black",
        linestyle="--",
        label="Runge function",
        zorder=3,
    )

    # Neural network predictions
    for a, activation_name in enumerate(results["activations"]):
        ax.plot(
            x_grid,
            results["grid_pred"][a],
            label=ACTIVATION_LABELS[str(activation_name)],
            zorder=2,
        )

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_xlim(-1, 1)
    ax.grid()
    legend_below(ax, ncol=3)
    save_figure(fig, "activation_regression_fit.pdf", fig_dir=fig_dir)


# -------------------------------------------------------------
# Run and plot Task C
# -------------------------------------------------------------


def plot(cfg=None) -> None:
    """
    Plot the saved Task C results.
    """
    fig_dir, result_dir = make_output_dirs(TASK)
    results = load_results(result_dir / RESULTS_NAME)

    plot_architecture_heatmaps(results, fig_dir)
    plot_depth_study(results, fig_dir)
    plot_width_study(results, fig_dir)
    plot_overfitting(results, fig_dir)
    plot_loss_curves(results, fig_dir)
    plot_regression_fit(results, fig_dir)


def run(cfg: dict) -> None:
    """
    Run Task C: save the results and generate figures. We:

    - Compare activations across network depths and widths.
    - Save training MSE, validation MSE and loss histories.
    - Print the best architectures.
    """
    _, result_dir = make_output_dirs(TASK)
    study = cfg["activation_study"]

    results = {
        "activations": np.array(study["activations"]),
        "depths": np.array(study["depths"]),
        "widths": np.array(study["widths"]),
        "optimizer": study["optimizer"],
        "learning_rate": study["learning_rate"],
        "epochs": cfg["training"]["epochs"],
    }

    results.update(run_activation_architecture_study(cfg))

    save_results(results, result_dir / RESULTS_NAME)
    print_summary(results)
    plot(cfg)
