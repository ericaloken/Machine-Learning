# classification.py
"""
Task e: classification of handwritten digits (MNIST) with a feed-forward
neural network, compared with logistic regression.

The output layer is Softmax with cross-entropy loss. A network without hidden
layers (layers = [784, 10]) is multinomial logistic regression, which is used
as the baseline.

We do a grid search over activation function, depth, width, learning rate,
regularizer (L1/L2) and regularization strength, choose the combination with
the highest validation accuracy and evaluate it once on the test set.

Main functions
--------------
run()   Train every combination in parallel, select the best on validation
        data, evaluate it on the test set, save results to classification.npz
        and make the figures by calling plot().
plot()  Load classification.npz and make the figures, so figures can be
        changed without retraining.

Pipeline
--------
1. Data load_data()
    - MNIST, flattened to 784 pixels and scaled to [0, 1]
    - the first n_train training images, split into train and validation
      (stratified, validation_size from [data])
    - train: fit the weights
    - validation: select the best combination
    - test: the separate 10 000 MNIST test images, used once for the chosen model

2. Grid search run()
    - every combination of activation, depth, width, learning rate,
      regularizer and lambda, with layers = [784] + depth * [width] + [10]
    - duplicates are skipped: depth 0 does not depend on activation or width,
      and lambda = 0 is the same model for L1 and L2 (stored under the first
      regularizer)
    - all runs are trained in parallel with joblib, n_jobs from [parallel]
    - record train and validation accuracy for each run

3. Final model run()
    - retrain the combination with the highest validation accuracy
    - test accuracy and confusion matrix on the test set

Saved results (output/classification/results/classification.npz)
-----------------------------------------------------------------
Settings:  activations, depths, widths, gammas, regularizers, lambdas
Per run, with A = n_activations, D = n_depths, W = n_widths,
G = n_learning_rates, R = n_regularizers, L = n_lambdas:
    train_acc    (A, D, W, G, R, L)   training accuracy, nan if not trained
    val_acc      (A, D, W, G, R, L)   validation accuracy, nan if not trained
For the chosen combination:
    best_index   (6,)                 index into the arrays above
    history      (epochs,)            training loss per epoch
    test_acc     ()                   test accuracy
    confusion    (10, 10)             test confusion matrix, rows true digit
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator, PercentFormatter
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from activations import Sigmoid, ReLU, LeakyReLU, Softmax
from losses import CrossEntropy
from regularizers import L1, L2
from network import build_network
from optimizers import make_optimizer
from data import load_mnist, MNIST_URL
from metrics import accuracy, confusion_matrix, precision_recall_f1
from experiments.common import make_output_dirs, save_results, load_results
from plotting_utils import save_figure, scientific_label, legend_below, HEATMAP_CMAP, mark_zeros, highlight_cell
from itertools import product
from pathlib import Path

TASK = "classification"
RESULTS_NAME = "classification.npz"
ACTIVATIONS = {"sigmoid": Sigmoid, "relu": ReLU, "leaky_relu": LeakyReLU}
ACTIVATION_LABELS = {
    "sigmoid": "Sigmoid",
    "relu": "ReLU",
    "leaky_relu": "LReLU",
}  # for plot titles
REGULARIZER_LABELS = {"l1": r"$\ell_1$", "l2": r"$\ell_2$"}  # for plot text, as in the report
REGULARIZERS = {"l1": L1, "l2": L2}


# ------------------------------------------
# Data
# ------------------------------------------


def load_data(cfg) -> dict:
    """
    Load MNIST with a stratified validation split taken from the training set only.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary. Uses data_path and n_train from
        [classification], and seed and validation_size from [data].

    Returns
    -------
    dict
        {"train": (X, y), "val": (X, y), "test": (X, y)}, with X of shape
        (n, 784) scaled to [0, 1] and y the digit labels 0-9.
    """
    c, seed = cfg["classification"], cfg["data"]["seed"]
    X, y, X_test, y_test = load_mnist(c["data_path"])
    X, y = X[: c["n_train"]], y[: c["n_train"]]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=cfg["data"]["validation_size"], stratify=y, random_state=seed
    )
    return {"train": (X_tr, y_tr), "val": (X_val, y_val), "test": (X_test, y_test)}


def one_hot(y, n_classes=10):
    """Labels (n,) to one-hot targets (n, n_classes), e.g. 3 -> [0, 0, 0, 1, 0, ...]."""
    return np.eye(n_classes)[y]


# ------------------------------------------
# Training and evaluation
# ------------------------------------------


def train_classifier(cfg, layers, activation_name, gamma, reg_name, lmbda, X, y):
    """
    Train a classifier with Softmax output and cross-entropy loss.
    layers = [784, 10] gives logistic regression.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary. Uses optimizer, epochs and batch_size from
        [classification], and seed from [data].
    layers : list[int]
        Network architecture, including input and output layers.
    activation_name : str
        Hidden-layer activation function, a key in ACTIVATIONS.
    gamma : float
        Learning rate.
    reg_name : str
        Type of regularization, "l1" or "l2". Ignored when lmbda = 0.
    lmbda : float
        Regularization strength. 0 gives no regularization.
    X : np.ndarray
        Training images, shape (n, 784).
    y : np.ndarray
        Training labels 0-9, shape (n,).

    Returns
    -------
    net : NeuralNetwork
        Trained neural network.
    history : np.ndarray
        Training loss (cross-entropy, without the penalty) after each epoch.
    """
    c = cfg["classification"]
    hidden = ACTIVATIONS[activation_name]
    regularizer = REGULARIZERS[reg_name](lmbda) if lmbda > 0 else None
    activations = [hidden() for _ in layers[1:-1]] + [Softmax()]
    net = build_network(
        layers,
        activations,
        CrossEntropy(),
        make_optimizer(c["optimizer"], gamma),
        regularizer=regularizer,
        random_state=cfg["data"]["seed"],
    )
    history = net.fit(X, one_hot(y), epochs=c["epochs"], batch_size=c["batch_size"])
    return net, np.asarray(history, dtype=float)


def predict_labels(net, X):
    """Predicted digit for each image: the class with the highest Softmax probability."""
    return np.argmax(net.predict(X), axis=1)


def train_and_score(cfg, data, layers, activation_name, gamma, reg_name, lmbda):
    """
    Train one combination and compute training and validation accuracy.
    This is the function run() calls in parallel, once per combination.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.
    data : dict
        Training, validation and test datasets from load_data().
    layers : list[int]
        Network architecture, including input and output layers.
    activation_name : str
        Hidden-layer activation function.
    gamma : float
        Learning rate.
    reg_name : str
        Type of regularization, "l1" or "l2".
    lmbda : float
        Regularization strength.

    Returns
    -------
    train_acc : float
        Accuracy on the training data.
    val_acc : float
        Accuracy on the validation data.
    """
    net, history = train_classifier(
        cfg, layers, activation_name, gamma, reg_name, lmbda, *data["train"]
    )
    X_tr, y_tr = data["train"]
    X_val, y_val = data["val"]
    train_acc = accuracy(y_tr, predict_labels(net, X_tr))
    val_acc = accuracy(y_val, predict_labels(net, X_val))
    return train_acc, val_acc


# ------------------------------------------
# Grid search
# ------------------------------------------


def run(cfg) -> None:
    """
    Run Task e: grid search over activation, depth, width, learning rate,
    regularizer and lambda. We:

    - Train every combination in parallel and record train and validation accuracy.
    - Select the combination with the highest validation accuracy.
    - Retrain it and evaluate it once on the test set.
    - Save the results and make the figures.
    """
    c = cfg["classification"]
    data_path = Path(__file__).parent.parent / c["data_path"]
    data = load_data(cfg)

    # settings from [classification] in config.toml
    gammas = c["learning_rates"]
    lambdas = c["lambdas"]
    activations = c["activations"]
    widths = c["widths"]
    depths = c["depths"]
    n_in, n_out = c["input_output_layers"]
    regulators = c["regularizers"]

    # every combination, as (index, value) pairs: the index is the position in
    # the result arrays, the value is what the network is trained with
    combos = list(
        product(
            enumerate(activations),
            enumerate(depths),
            enumerate(widths),
            enumerate(gammas),
            enumerate(regulators),
            enumerate(lambdas),
        )
    )

    # one axis per setting, nan = not trained (duplicate)
    shape = (
        len(activations),
        len(depths),
        len(widths),
        len(gammas),
        len(regulators),
        len(lambdas),
    )
    train_acc = np.full(shape, np.nan)
    val_acc = np.full(shape, np.nan)

    # list of runs to do, duplicates skipped before any training
    todo = []
    for (
        (a, act),
        (d, depth),
        (w, width),
        (g, gamma),
        (r, reg_name),
        (l, lmbda),
    ) in combos:
        if depth == 0 and (
            a > 0 or w > 0
        ):  # no hidden layer: activation and width do not matter
            continue
        if lmbda == 0 and r > 0:  # no regularization: l1 and l2 are the same model
            continue
        layers = [n_in] + depth * [width] + [n_out]
        todo.append(((a, d, w, g, r, l), layers, act, gamma, reg_name, lmbda))

    # train all runs in parallel, results come back in the same order as todo
    runs = Parallel(n_jobs=cfg["parallel"]["n_jobs"], verbose=10)(
        delayed(train_and_score)(cfg, data, layers, act, gamma, reg_name, lmbda)
        for idx, layers, act, gamma, reg_name, lmbda in todo
    )

    # put each result in its cell
    for (idx, layers, act, gamma, reg_name, lmbda), (tr, va) in zip(todo, runs):
        train_acc[idx], val_acc[idx] = tr, va
        print(
            f"{act:>10} {str(layers):>16} gamma={gamma:.0e} "
            f"{reg_name} lambda={lmbda:g}: train={tr:.4f} val={va:.4f}"
        )

    # best combination on validation data
    a, d, w, g, r, l = np.unravel_index(np.nanargmax(val_acc), shape)
    best_layers = [n_in] + depths[d] * [widths[w]] + [n_out]
    print(
        f"\nBest: {activations[a]} {best_layers} gamma={gammas[g]:.0e} "
        f"{regulators[r]} lambda={lambdas[l]:g}, val accuracy = {val_acc[a, d, w, g, r, l]:.4f}"
    )

    # retrain best model and evaluate on the test set (only once, at the end)
    net, history = train_classifier(
        cfg,
        best_layers,
        activations[a],
        gammas[g],
        regulators[r],
        lambdas[l],
        *data["train"],
    )
    X_test, y_test = data["test"]
    y_pred = predict_labels(net, X_test)
    test_acc = accuracy(y_test, y_pred)
    print(f"Test accuracy = {test_acc:.4f}")

    # save so plot() can run without retraining
    write_to_file(
        cfg, train_acc, val_acc, (a, d, w, g, r, l), history, test_acc, y_test, y_pred
    )
    plot(cfg)


# ------------------------------------------
# Saving results
# ------------------------------------------


def write_to_file(
    cfg, train_acc, val_acc, best_index, history, test_acc, y_test, y_pred
) -> None:
    """
    Save the grid search results to output/classification/results/classification.npz,
    so plot() can run without retraining. See the module docstring for the contents.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary, the settings lists are saved with the results.
    train_acc, val_acc : np.ndarray
        Training and validation accuracy, shape (A, D, W, G, R, L).
    best_index : tuple[int]
        Index (a, d, w, g, r, l) of the chosen combination.
    history : np.ndarray
        Training loss per epoch for the chosen combination.
    test_acc : float
        Test accuracy of the chosen combination.
    y_test, y_pred : np.ndarray
        True and predicted test labels, used for the confusion matrix.
    """
    c = cfg["classification"]
    activations, depths, widths = c["activations"], c["depths"], c["widths"]
    gammas, regulators, lambdas = c["learning_rates"], c["regularizers"], c["lambdas"]

    _, result_dir = make_output_dirs(TASK)
    save_results(
        {
            "train_acc": train_acc,
            "val_acc": val_acc,
            "activations": np.array(activations),
            "depths": np.array(depths),
            "widths": np.array(widths),
            "gammas": np.array(gammas),
            "regularizers": np.array(regulators),
            "lambdas": np.array(lambdas),
            "best_index": np.array(best_index),
            "history": history,
            "test_acc": test_acc,
            "confusion": confusion_matrix(y_test, y_pred),
        },
        result_dir / RESULTS_NAME,
    )


# ------------------------------------------
# Plotting helpers
# ------------------------------------------


def highlight_best(ax, values):
    """Red box around the cell with the highest value (nan cells ignored)."""
    if np.isfinite(values).any():
        row, col = np.unravel_index(np.nanargmax(values), values.shape)
        highlight_cell(ax, col, row)


def best_over(values, axis):
    """Like np.nanmax, but gives nan (without a warning) where every value is nan."""
    filled = np.where(np.isnan(values), -np.inf, values)
    best = filled.max(axis=axis)
    return np.where(np.isneginf(best), np.nan, best)


def with_unregularized(values, lambdas):
    """
    lambda = 0 is only trained once, stored under the first regularizer. Copy it
    to the other regularizers, since it is the same unregularized model.
    Regularizer and lambda must be the last two axes.
    """
    values = values.copy()
    no_reg = lambdas == 0
    values[..., 1:, no_reg] = values[..., :1, no_reg]
    return values


def config_label(results, idx):
    """Short text label for one combination, e.g. 'ReLU [100, 100], $\\gamma$=10^-3, L2 $\\lambda$=1'."""
    a, d, w, g, r, l = idx
    depth, width = int(results["depths"][d]), int(results["widths"][w])
    lmbda = results["lambdas"][l]
    if depth == 0:
        model = "Logistic regression"
    else:
        model = f"{ACTIVATION_LABELS[str(results['activations'][a])]} {[width] * depth}"
    reg = (
        "no reg."
        if lmbda == 0
        else f"{REGULARIZER_LABELS[str(results['regularizers'][r])]} $\\lambda$={scientific_label(lmbda)}"
    )
    return f"{model}, $\\gamma$={scientific_label(results['gammas'][g])}, {reg}"


def gamma_lambda_heatmap(ax, grid, results, vmin, vmax):
    """Heatmap with lambda on the rows and learning rate on the columns, best cell boxed."""
    sns.heatmap(
        grid,
        ax=ax,
        cmap=HEATMAP_CMAP,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=".3f",
        annot_kws={"fontsize": 7},
        xticklabels=[scientific_label(g) for g in results["gammas"]],
        yticklabels=[scientific_label(lmbda) for lmbda in results["lambdas"]],
        cbar=False,
    )
    mark_zeros(ax, grid)
    highlight_best(ax, grid)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)


def shared_colorbar(fig, axes, vmin, vmax, label="Validation accuracy"):
    """One colorbar for several heatmaps that share vmin and vmax."""
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=HEATMAP_CMAP)
    cbar = fig.colorbar(sm, ax=axes, shrink=0.9, pad=0.02)
    cbar.set_label(label)


def figure_legend_below(fig, ax):
    """
    One legend centered below all panels of a multi-panel figure, with the
    entries from ax. For a single-panel figure, use legend_below from
    plotting_utils instead. The figure needs layout="constrained".
    """
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))


# ------------------------------------------
# Overview figures (output/classification/figures)
# ------------------------------------------


def plot_architecture_heatmaps(results, fig_dir) -> None:
    """
    Best validation accuracy for each depth and width, one heatmap per activation.
    Each cell is the best over learning rates and regularization. Depth 0
    (logistic regression) has no hidden layer, so it is left out here.
    """
    activations = results["activations"]
    depths, widths = results["depths"], results["widths"]
    hidden = depths > 0

    # best over learning rate, regularizer and lambda -> shape (A, D, W)
    best_acc = np.nanmax(results["val_acc"][:, hidden], axis=(3, 4, 5))
    vmin, vmax = np.nanmin(best_acc), np.nanmax(best_acc)
    print_best_per_architecture(results)

    fig, axes = plt.subplots(
        1, len(activations), figsize=(7.06, 2.8), sharey=True, layout="constrained"
    )
    axes = np.atleast_1d(axes)

    for a, (ax, activation_name) in enumerate(zip(axes, activations)):
        sns.heatmap(
            best_acc[a],
            ax=ax,
            cmap=HEATMAP_CMAP,
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".4f",
            xticklabels=widths,
            yticklabels=depths[hidden],
            cbar=(a == len(activations) - 1),
            cbar_kws={"label": "Validation accuracy"},
        )
        mark_zeros(ax, best_acc[a])
        highlight_best(ax, best_acc[a])
        ax.set_title(ACTIVATION_LABELS[activation_name])
        ax.set_xlabel("Neurons per hidden layer")
        ax.tick_params(axis="y", rotation=0)

    axes[0].set_ylabel("Number of hidden layers")
    save_figure(
        fig, "classification_architecture_heatmaps.pdf", fig_dir=fig_dir, Tight=False
    )


def print_best_per_architecture(results) -> None:
    """
    Print the best setting (learning rate, regularizer, lambda) and validation
    accuracy for every activation and architecture, the numbers shown in the
    architecture heatmaps. Logistic regression (depth 0) is the first row.
    """
    activations, depths, widths = results["activations"], results["depths"], results["widths"]
    gammas, regularizers, lambdas = results["gammas"], results["regularizers"], results["lambdas"]
    val = results["val_acc"]

    def row(name, layers, values):
        g, r, l = np.unravel_index(np.nanargmax(values), values.shape)
        reg = "--" if lambdas[l] == 0 else str(regularizers[r])  # lambda = 0: no regularization
        print(f"{name:>12} {layers:>12} {gammas[g]:>8g} {reg:>5} {lambdas[l]:>8g} {values[g, r, l]:>10.4f}")

    print("\nBest validation accuracy per architecture (classification_architecture_heatmaps.pdf)")
    print(f"{'Activation':>12} {'Layers':>12} {'gamma':>8} {'Reg.':>5} {'lambda':>8} {'Val. acc.':>10}")
    row("Logistic", "None", val[0, depths.tolist().index(0), 0])
    for a, activation_name in enumerate(activations):
        for d, depth in enumerate(depths):
            if depth == 0:
                continue
            for w, width in enumerate(widths):
                layers = str(int(depth) * [int(width)])
                row(ACTIVATION_LABELS[str(activation_name)], layers, val[a, d, w])
    print()


def plot_regularization_heatmaps(results, fig_dir) -> None:
    """
    Validation accuracy over learning rate and lambda for the best model's
    activation and architecture, one heatmap for L1 and one for L2.
    """
    a, d, w = results["best_index"][:3]
    gammas, lambdas = results["gammas"], results["lambdas"]
    regularizers = results["regularizers"]

    # shape (G, R, L). lambda = 0 is only trained once (stored under the first
    # regularizer), so copy it to the others: it is the same unregularized model
    values = results["val_acc"][a, d, w].copy()
    no_reg = lambdas == 0
    values[:, :, no_reg] = values[:, :1, no_reg]
    vmin, vmax = np.nanmin(values), np.nanmax(values)

    hidden = [int(results["widths"][w])] * int(results["depths"][d])
    fig, axes = plt.subplots(
        1, len(regularizers), figsize=(7.06, 3.0), sharey=True, layout="constrained"
    )
    axes = np.atleast_1d(axes)

    for r, (ax, reg_name) in enumerate(zip(axes, regularizers)):
        grid = values[:, r, :].T  # rows lambda, columns gamma
        sns.heatmap(
            grid,
            ax=ax,
            cmap=HEATMAP_CMAP,
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".3f",
            annot_kws={"fontsize": 8},
            xticklabels=[scientific_label(g) for g in gammas],
            yticklabels=[scientific_label(lmbda) for lmbda in lambdas],
            cbar=(r == len(regularizers) - 1),
            cbar_kws={"label": "Validation accuracy"},
        )
        mark_zeros(ax, grid)
        highlight_best(ax, grid)
        ax.set_title(f"{REGULARIZER_LABELS[str(reg_name)]}, hidden layers {hidden}")
        ax.set_xlabel(r"Learning rate $\gamma$")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)

    axes[0].set_ylabel(r"Regularization strength $\lambda$")
    save_figure(
        fig, "classification_regularization_heatmaps.pdf", fig_dir=fig_dir, Tight=False
    )


def plot_accuracy_vs_depth(results, fig_dir) -> None:
    """
    Best validation accuracy for each number of hidden layers, where depth 0
    is logistic regression. Shows how much the hidden layers add over the baseline.
    """
    depths = results["depths"]
    best_acc = [np.nanmax(results["val_acc"][:, d]) for d in range(len(depths))]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(depths, best_acc, marker="o")
    for depth, acc in zip(depths, best_acc):
        ax.annotate(
            f"{acc:.3f}",
            (depth, acc),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
        )

    ax.margins(y=0.2)  # room for the value labels above the points
    ax.set_xticks(depths)
    ax.set_xticklabels(
        ["0\n(logreg)" if depth == 0 else str(depth) for depth in depths]
    )
    ax.set_xlabel("Number of hidden layers")
    ax.set_ylabel("Best validation accuracy")
    save_figure(fig, "classification_accuracy_vs_depth.pdf", fig_dir=fig_dir)


def plot_loss_curve(results, fig_dir) -> None:
    """Training loss (cross-entropy) per epoch for the retrained best model."""
    history = results["history"]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(np.arange(1, len(history) + 1), history)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    save_figure(fig, "classification_loss_curve.pdf", fig_dir=fig_dir)


def plot_confusion_matrix(results, fig_dir, log=False) -> None:
    """
    Confusion matrix of the best model on the test set, as the fraction of each
    true digit, so every row sums to 1.

    log=False gives a linear color scale. log=True gives a logarithmic one, so that
    small off-diagonal values are not washed out by a large diagonal.
    """
    C = results["confusion"]
    fraction = C / C.sum(axis=1, keepdims=True)

    # per digit, and the macro average: mean over the 10 digits, every digit counts equally
    precision, recall, f1 = precision_recall_f1(C)
    print(f"\nTest set\n{'Digit':>6} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for k in range(len(f1)):
        print(f"{k:>6} {precision[k]:>10.4f} {recall[k]:>10.4f} {f1[k]:>10.4f}")
    print(f"{'Mean':>6} {precision.mean():>10.4f} {recall.mean():>10.4f} {f1.mean():>10.4f}\n")

    # exact zeros have no logarithm, they are drawn in ZERO_COLOR by mark_zeros either way
    norm = LogNorm(vmin=fraction[fraction > 0].min(), vmax=1) if log else None

    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    sns.heatmap(
        fraction,
        ax=ax,
        cmap=HEATMAP_CMAP,
        norm=norm,
        annot=True,
        fmt=".3f",
        annot_kws={"fontsize": 5.5},
    )
    mark_zeros(ax, fraction)
    if log:
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0.001, 0.01, 0.1, 1])
        cbar.set_ticklabels(["0.001", "0.01", "0.1", "1"])
    # ax.set_title(f"Test accuracy = {float(results['test_acc']):.4f}, macro F1 = {f1.mean():.4f}")
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.tick_params(axis="y", rotation=0)
    save_figure(fig, "classification_confusion_matrix.pdf", fig_dir=fig_dir)


def plot_precision_recall_f1(results, fig_dir) -> None:
    """
    Precision, recall and F1 per digit for the best model on the test set, from the
    saved confusion matrix, shown as 1 - score so the bars start at 0 and small
    differences between scores close to 1 are visible. 1 - precision is the share
    of predictions of a digit that are wrong, 1 - recall the share of a digit that
    is missed. The dashed line is 1 - mean F1 (macro average).
    """
    precision, recall, f1 = precision_recall_f1(results["confusion"])
    digits = np.arange(len(f1))

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    width = 0.27
    ax.bar(digits - width, 1 - precision, width, color="C0", label=r"$1 - \mathrm{Precision}$")
    ax.bar(digits, 1 - recall, width, color="C1", label=r"$1 - \mathrm{Recall}$")
    ax.bar(digits + width, 1 - f1, width, color="C2", label=r"$1 - \mathrm{F1}$")
    ax.axhline(
        1 - f1.mean(), color="gray", linestyle="--", linewidth=1,
        label=f"$1 - \\mathrm{{mean\\ F1}} = {100 * (1 - f1.mean()):.2f}\\%$",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax.set_xticks(digits)
    ax.set_xlabel("Digit")
    ax.set_ylabel(r"$1 - \mathrm{score}$")
    ax.set_title("Precision, recall and F1 per digit (test set)")
    legend_below(ax, ncol=2)
    save_figure(fig, "classification_precision_recall_f1.pdf", fig_dir=fig_dir)


# ------------------------------------------
# Every combination (output/classification/figures/all_combinations)
# ------------------------------------------


def plot_all_combinations(results, fig_dir) -> None:
    """
    Every trained combination. One figure per activation and regularizer, with
    one learning rate x lambda heatmap per architecture (rows depth, columns
    width). Logistic regression (depth 0) gets its own figure per regularizer.
    """
    depths, widths = results["depths"], results["widths"]
    values = with_unregularized(results["val_acc"], results["lambdas"])

    for r, reg_name in enumerate(results["regularizers"]):
        # neural networks
        for a, activation_name in enumerate(results["activations"]):
            hidden_d = [d for d, depth in enumerate(depths) if depth > 0]
            if not hidden_d:
                continue
            block = values[a, hidden_d][..., r, :]  # (D, W, G, L)
            vmin, vmax = np.nanmin(block), np.nanmax(block)

            fig, axes = plt.subplots(
                len(hidden_d),
                len(widths),
                figsize=(7.06, 0.8 + 2.4 * len(hidden_d)),
                layout="constrained",
                squeeze=False,
            )
            for i, d in enumerate(hidden_d):
                for w, width in enumerate(widths):
                    ax = axes[i, w]
                    gamma_lambda_heatmap(
                        ax, values[a, d, w, :, r, :].T, results, vmin, vmax
                    )
                    ax.set_title(f"Hidden layers {[int(width)] * int(depths[d])}")
                    ax.set_xlabel(
                        r"Learning rate $\gamma$" if i == len(hidden_d) - 1 else ""
                    )
                    ax.set_ylabel(r"$\lambda$" if w == 0 else "")
            shared_colorbar(fig, axes, vmin, vmax)
            fig.suptitle(
                f"{ACTIVATION_LABELS[str(activation_name)]}, {REGULARIZER_LABELS[str(reg_name)]} regularization"
            )
            save_figure(
                fig,
                f"all_combinations_{activation_name}_{reg_name}.pdf",
                fig_dir=fig_dir,
                Tight=False,
            )

        # logistic regression
        if 0 in depths:
            d0 = int(np.flatnonzero(depths == 0)[0])
            grid = values[0, d0, 0, :, r, :].T
            vmin, vmax = np.nanmin(grid), np.nanmax(grid)
            fig, ax = plt.subplots(figsize=(4.0, 3.2), layout="constrained")
            gamma_lambda_heatmap(ax, grid, results, vmin, vmax)
            shared_colorbar(fig, ax, vmin, vmax)
            ax.set_title(f"Logistic regression, {REGULARIZER_LABELS[str(reg_name)]} regularization")
            ax.set_xlabel(r"Learning rate $\gamma$")
            ax.set_ylabel(r"$\lambda$")
            save_figure(
                fig,
                f"all_combinations_logreg_{reg_name}.pdf",
                fig_dir=fig_dir,
                Tight=False,
            )


# ------------------------------------------
# Overview figures, continued
# ------------------------------------------


def plot_overfitting_gap(results, fig_dir) -> None:
    """
    Train accuracy minus validation accuracy for each architecture, using the
    learning rate and regularization that gave the best validation accuracy.
    A large gap means the network memorises the training data.
    """
    activations, depths, widths = (
        results["activations"],
        results["depths"],
        results["widths"],
    )
    hidden = depths > 0
    val = results["val_acc"][:, hidden]
    train = results["train_acc"][:, hidden]

    # best (gamma, regularizer, lambda) per architecture
    shape = val.shape[:3] + (-1,)
    best = np.argmax(np.where(np.isnan(val), -np.inf, val).reshape(shape), axis=-1)[
        ..., None
    ]
    gap = (
        np.take_along_axis(train.reshape(shape), best, axis=-1)
        - np.take_along_axis(val.reshape(shape), best, axis=-1)
    )[..., 0]
    vmin, vmax = np.nanmin(gap), np.nanmax(gap)

    fig, axes = plt.subplots(
        1, len(activations), figsize=(7.06, 2.8), sharey=True, layout="constrained"
    )
    axes = np.atleast_1d(axes)
    for a, (ax, activation_name) in enumerate(zip(axes, activations)):
        sns.heatmap(
            gap[a],
            ax=ax,
            cmap=HEATMAP_CMAP,
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".3f",
            xticklabels=widths,
            yticklabels=depths[hidden],
            cbar=False,
        )
        mark_zeros(ax, gap[a])
        ax.set_title(ACTIVATION_LABELS[str(activation_name)])
        ax.set_xlabel("Neurons per hidden layer")
        ax.tick_params(axis="y", rotation=0)
    axes[0].set_ylabel("Number of hidden layers")
    shared_colorbar(fig, axes, vmin, vmax, label="Train $-$ validation accuracy")
    save_figure(fig, "classification_overfitting_gap.pdf", fig_dir=fig_dir, Tight=False)


def plot_accuracy_vs_learning_rate(results, fig_dir) -> None:
    """
    Best validation accuracy as a function of learning rate, one panel per number
    of hidden layers, one line per activation (best over width and regularization).
    Logistic regression is the grey dashed reference line.
    """
    activations, depths, gammas = (
        results["activations"],
        results["depths"],
        results["gammas"],
    )
    val = results["val_acc"]
    hidden_d = [d for d, depth in enumerate(depths) if depth > 0]

    fig, axes = plt.subplots(
        1,
        len(hidden_d),
        figsize=(7.06, 3.0),
        sharey=True,
        layout="constrained",
        squeeze=False,
    )
    for ax, d in zip(axes[0], hidden_d):
        for a, activation_name in enumerate(activations):
            ax.plot(
                gammas,
                best_over(val[a, d], axis=(0, 2, 3)),
                marker="o",
                color=f"C{a}",
                label=ACTIVATION_LABELS[str(activation_name)],
            )
        if 0 in depths:
            d0 = int(np.flatnonzero(depths == 0)[0])
            ax.plot(
                gammas,
                best_over(val[0, d0, 0], axis=(1, 2)),
                marker="o",
                color="gray",
                linestyle="--",
                label="Logistic regression",
            )
        ax.set_xscale("log")
        ax.set_title(f"{int(depths[d])} hidden layer{'s' if depths[d] > 1 else ''}")
        ax.set_xlabel(r"Learning rate $\gamma$")
    axes[0, 0].set_ylabel("Best validation accuracy")
    figure_legend_below(fig, axes[0, 0])
    save_figure(
        fig,
        "classification_accuracy_vs_learning_rate.pdf",
        fig_dir=fig_dir,
        Tight=False,
    )


def plot_accuracy_vs_lambda(results, fig_dir) -> None:
    """
    Best validation accuracy as a function of lambda, one panel per regularizer,
    one line per activation (best over architecture and learning rate).
    Logistic regression is the grey dashed reference line.
    """
    activations, depths, lambdas = (
        results["activations"],
        results["depths"],
        results["lambdas"],
    )
    val = with_unregularized(results["val_acc"], lambdas)
    hidden = depths > 0
    x = np.arange(
        len(lambdas)
    )  # evenly spaced, since lambda = 0 cannot go on a log axis

    fig, axes = plt.subplots(
        1,
        len(results["regularizers"]),
        figsize=(7.06, 3.0),
        sharey=True,
        layout="constrained",
        squeeze=False,
    )
    for r, (ax, reg_name) in enumerate(zip(axes[0], results["regularizers"])):
        for a, activation_name in enumerate(activations):
            ax.plot(
                x,
                best_over(val[a, hidden], axis=(0, 1, 2))[r],
                marker="o",
                color=f"C{a}",
                label=ACTIVATION_LABELS[str(activation_name)],
            )
        if 0 in depths:
            d0 = int(np.flatnonzero(depths == 0)[0])
            ax.plot(
                x,
                best_over(val[0, d0, 0], axis=0)[r],
                marker="o",
                color="gray",
                linestyle="--",
                label="Logistic regression",
            )
        ax.set_xticks(x)
        ax.set_xticklabels([scientific_label(lmbda) for lmbda in lambdas])
        ax.set_title(f"{REGULARIZER_LABELS[str(reg_name)]} regularization")
        ax.set_xlabel(r"Regularization strength $\lambda$")
    axes[0, 0].set_ylabel("Best validation accuracy")
    figure_legend_below(fig, axes[0, 0])
    save_figure(
        fig, "classification_accuracy_vs_lambda.pdf", fig_dir=fig_dir, Tight=False
    )


def plot_top_configurations(results, fig_dir, n_top=15) -> None:
    """
    The n_top combinations with the highest validation accuracy, with their
    training accuracy next to it. Best at the top.
    """
    val, train = results["val_acc"], results["train_acc"]
    flat = np.where(np.isnan(val), -np.inf, val).ravel()
    order = np.argsort(-flat)[:n_top]
    order = order[np.isfinite(flat[order])]
    idxs = [np.unravel_index(i, val.shape) for i in order]

    y = np.arange(len(idxs))
    val_top = np.array([val[idx] for idx in idxs])
    train_top = np.array([train[idx] for idx in idxs])

    fig, ax = plt.subplots(figsize=(7.06, 0.8 + 0.28 * len(idxs)))
    ax.hlines(y, val_top, train_top, color="lightgray", linewidth=2, zorder=1)
    ax.scatter(val_top, y, color="C0", s=30, zorder=2, label="Validation")
    ax.scatter(train_top, y, color="C1", marker="s", s=30, zorder=2, label="Train")
    ax.set_yticks(y)
    ax.set_yticklabels([config_label(results, idx) for idx in idxs], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy")
    ax.set_title(f"Top {len(idxs)} combinations by validation accuracy")
    legend_below(ax, ncol=2)
    save_figure(fig, "classification_top_configurations.pdf", fig_dir=fig_dir)


# ------------------------------------------
# Plot entry point
# ------------------------------------------


def plot(cfg=None) -> None:
    """
    Plot the saved Task e results, no training needed.
    Run with: python main.py classification plot
    """
    fig_dir, result_dir = make_output_dirs(TASK)
    results = load_results(result_dir / RESULTS_NAME)

    # overview
    plot_architecture_heatmaps(results, fig_dir)
    plot_regularization_heatmaps(results, fig_dir)
    plot_accuracy_vs_depth(results, fig_dir)
    plot_accuracy_vs_learning_rate(results, fig_dir)
    plot_accuracy_vs_lambda(results, fig_dir)
    plot_overfitting_gap(results, fig_dir)
    plot_top_configurations(results, fig_dir)
    plot_loss_curve(results, fig_dir)
    plot_confusion_matrix(results, fig_dir, log=True)  # log=False for a linear color scale
    plot_precision_recall_f1(results, fig_dir)

    # every combination, in its own folder so the overview stays readable
    plot_all_combinations(results, fig_dir / "all_combinations")
