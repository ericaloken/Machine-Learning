# cross_calidation.py

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import clone
import numpy as np
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import seaborn as sns

from plotting_utils import scientific_label, save_figure, HEATMAP_CMAP, mark_zeros, highlight_cell

from data import load_mnist
from metrics import accuracy
from experiments.classification import train_classifier, predict_labels
from sklearn_estimators import NeuralNetRegressor
from experiments.common import (
    get_runge_data,
    make_output_dirs,
    save_results,
    load_results,
)


def run_runge(cfg):
    """
    Perform k-fold cross-validation for Runge function regression.

    Evaluates combinations of learning rates and regularization strengths
    using a neural network regressor. Selects the best configuration and
    applies the one-standard-error rule before evaluating on test data.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing dataset, network,
        training, cross-validation, and parallelization settings.

    Returns
    -------
    None
        Results are printed and saved to an NPZ file.
    """
    c = cfg["cross_validation"]

    X_train, X_test, y_train, y_test = get_runge_data(cfg, n_points=c["n_points"])

    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

    cv = KFold(n_splits=c["n_folds"], shuffle=True, random_state=cfg["data"]["seed"])

    net = make_pipeline(
        StandardScaler(),
        NeuralNetRegressor(
            hidden=tuple(c["architecture"]),
            activation=c["activation"],
            optimizer=c["optimizer"],
            regularizer=c["regularizer"],
            epochs=cfg["training"]["epochs"],
            batch_size=cfg["training"]["batch_size"],
            seed=cfg["data"]["seed"],
        ),
    )

    # Standardise y within each training fold
    net = TransformedTargetRegressor(regressor=net, transformer=StandardScaler())

    param_grid = {
        "regressor__neuralnetregressor__gamma": c["learning_rates"],
        "regressor__neuralnetregressor__lmbd": c["lambdas"],
    }

    search = GridSearchCV(
        net,
        param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        return_train_score=True,
        n_jobs=cfg["parallel"]["n_jobs"],
        refit=True,
    )

    search.fit(X_train, y_train)

    # Extract CV results
    results = search.cv_results_

    gammas = np.asarray(c["learning_rates"])
    lambdas = np.asarray(c["lambdas"])

    shape = (len(gammas), len(lambdas))

    train_mse = -results["mean_train_score"].reshape(shape)
    cv_mse = -results["mean_test_score"].reshape(shape)

    # Standard error of fold errors
    k = cv.get_n_splits()
    cv_se = results["std_test_score"].reshape(shape) / np.sqrt(k - 1)

    # Best CV parameters
    best = np.unravel_index(np.argmin(cv_mse), cv_mse.shape)

    best_gamma = gammas[best[0]]
    best_lambda = lambdas[best[1]]

    best_mse = cv_mse[best]
    best_se = cv_se[best]

    # One-standard-error threshold
    threshold = best_mse + best_se

    # All configurations within one SE
    within = cv_mse <= threshold

    # Select the strongest regularization within one SE
    candidates = np.argwhere(within)

    one_se = max(candidates, key=lambda ij: (lambdas[ij[1]], -cv_mse[tuple(ij)]))

    one_se = tuple(one_se)

    one_se_gamma = gammas[one_se[0]]
    one_se_lambda = lambdas[one_se[1]]

    print("\nMinimum CV MSE:")
    print(f"  gamma  = {best_gamma:.4g}")
    print(f"  lambda = {best_lambda:.4g}")
    print(f"  MSE    = {best_mse:.6f} ± {best_se:.6f}")

    print("\nOne-SE selected parameters:")
    print(f"  gamma  = {one_se_gamma:.4g}")
    print(f"  lambda = {one_se_lambda:.4g}")
    print(f"  MSE    = {cv_mse[one_se]:.6f}")

    print(f"\nOne-SE threshold: {threshold:.6f}")

    # Refit the model selected by the one-SE rule
    one_se_model = clone(search.estimator)

    one_se_model.set_params(
        regressor__neuralnetregressor__gamma=float(one_se_gamma),
        regressor__neuralnetregressor__lmbd=float(one_se_lambda),
    )

    # Train on the entire training dataset
    one_se_model.fit(X_train, y_train)

    # Evaluate on the independent test dataset
    y_pred = one_se_model.predict(X_test)

    test_mse = mean_squared_error(y_test, y_pred)
    print(f"One-SE test MSE: {test_mse:.6f}")

    # Save numerical results
    _, result_dir = make_output_dirs("cross_validation")

    save_results(
        {
            "gammas": gammas,
            "lambdas": lambdas,
            "train_mse": train_mse,
            "cv_mse": cv_mse,
            "cv_se": cv_se,
            "best_index": search.best_index_,
            "test_mse": test_mse,
            "one_se_index": np.asarray(one_se),
            "one_se_threshold": threshold,
        },
        result_dir / "cross_validation.npz",
    )


def plot_runge(cfg=None):
    """
    Plot training and cross-validation MSE for Runge regression.

    Creates two heatmaps showing the MSE for different learning rates
    and regularization strengths. Marks the minimum MSE and the
    one-standard-error selection.

    Parameters
    ----------
    cfg : dict or None, optional
        Configuration dictionary. Currently unused.

    Returns
    -------
    None
        Saves the figure as a PDF.
    """
    fig_dir, result_dir = make_output_dirs("cross_validation")

    results = load_results(result_dir / "cross_validation.npz")

    gammas = results["gammas"]
    lambdas = results["lambdas"]

    train_mse = results["train_mse"]
    cv_mse = results["cv_mse"]
    cv_se = results["cv_se"]

    best = np.unravel_index(int(results["best_index"]), cv_mse.shape)

    within = cv_mse <= cv_mse[best] + cv_se[best]
    one_se = tuple(results["one_se_index"].astype(int))

    # Figure layout: two heatmaps and one shared colorbar
    fig = plt.figure(figsize=(7.06, 3.0))

    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.08)

    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    cbar_ax = fig.add_subplot(gs[0, 2])

    # Shared color scale
    vmin = np.nanmin([np.nanmin(train_mse), np.nanmin(cv_mse)])
    vmax = np.nanmax([np.nanmax(train_mse), np.nanmax(cv_mse)])

    for i, (ax, data, title) in enumerate(
        zip(axes, [train_mse, cv_mse], ["Training MSE", "Cross-validation MSE"])
    ):
        sns.heatmap(
            data.T,
            annot=True,
            fmt=".4f",
            cmap=HEATMAP_CMAP,
            annot_kws={"fontsize": 8},
            mask=~np.isfinite(data.T),
            xticklabels=[scientific_label(g) for g in gammas],
            yticklabels=[scientific_label(l) for l in lambdas],
            vmin=vmin,
            vmax=vmax,
            cbar=(i == 1),
            cbar_ax=cbar_ax if i == 1 else None,
            cbar_kws={"label": "MSE"},
            ax=ax,
        )
        mark_zeros(ax, data.T)

        ax.set_xlabel(r"Learning rate $\gamma$")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)
        ax.set_title(title)

        if i == 0:
            ax.set_ylabel(r"Regularization $\lambda$")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", left=False, labelleft=False)

    # Everything below is OUTSIDE the heatmap loop

    # Mark cells within one standard error on CV heatmap
    for i, j in np.argwhere(within):
        # black star with white edge, visible on both light and dark cells
        axes[1].plot(
            i + 0.85, j + 0.22, marker="*", markersize=7,
            markerfacecolor="black", markeredgecolor="white", markeredgewidth=0.5,
        )

    # Best training MSE
    train_best = np.unravel_index(np.nanargmin(train_mse), train_mse.shape)

    # Mark training minimum (left heatmap)
    highlight_cell(axes[0], *train_best)

    # Mark CV minimum (right heatmap)
    highlight_cell(axes[1], *best)

    # Mark one-SE selection (right heatmap only)
    highlight_cell(axes[1], *one_se, color="C2", linestyle="--")

    save_figure(fig, "cross_validation_heatmap.pdf", fig_dir=fig_dir)


# MNIST


def load_mnist_settings(cfg):
    """
    Load the best MNIST classifier settings from previous experiments.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing classification settings.

    Returns
    -------
    layers : list of int
        Neural network architecture, including input and output layers.
    activation_name : str
        Selected hidden-layer activation function.
    regularizer_name : str
        Selected regularization method.
    optimizer_name : str
        Optimizer specified in the classification configuration.
    """
    _, result_dir = make_output_dirs("classification")
    result_path = result_dir / "classification.npz"
    if not result_path.exists():
        raise FileNotFoundError(
            f"{result_path} not found. Run the classification experiment first "
            "to save its selected configuration."
        )

    res = load_results(result_path)
    a, d, w, _, r, _ = map(int, res["best_index"])
    depth = int(res["depths"][d])
    width = int(res["widths"][w])
    n_in, n_out = map(int, cfg["classification"]["input_output_layers"])
    layers = [n_in] + [width] * depth + [n_out]
    activation_name = str(res["activations"][a])
    regularizer_name = str(res["regularizers"][r])
    optimizer_name = cfg["classification"]["optimizer"]
    return layers, activation_name, regularizer_name, optimizer_name


def selected_mnist_settings(cfg):
    """
    Retrieve fixed MNIST classifier settings from the configuration.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing MNIST cross-validation settings.

    Returns
    -------
    layers : list of int
        Neural network architecture.
    activation_name : str
        Hidden-layer activation function.
    regularizer_name : str
        Regularization method.
    optimizer_name : str
        Optimization algorithm.
    """

    c = cfg["mnist_cv"]

    layers = c["architecture"]
    activation_name = c["activation"]
    regularizer_name = c["regularizer"]
    optimizer_name = c["optimizer"]

    return layers, activation_name, regularizer_name, optimizer_name


def train_cv_fold(
    train_cfg, layers, activation_name, reg_name, gamma, lmbda, X, y, train_idx, val_idx
):
    """
    Train and evaluate a classifier on one cross-validation fold.

    Parameters
    ----------
    train_cfg : dict
        Configuration dictionary containing training settings.
    layers : list of int
        Neural network architecture.
    activation_name : str
        Hidden-layer activation function.
    reg_name : str
        Regularization method.
    gamma : float
        Learning rate.
    lmbda : float
        Regularization strength.
    X : ndarray
        Input data.
    y : ndarray
        Target class labels.
    train_idx : array_like
        Indices of samples used for training.
    val_idx : array_like
        Indices of samples used for validation.

    Returns
    -------
    train_acc : float
        Classification accuracy on the training fold.
    val_acc : float
        Classification accuracy on the validation fold.
    """

    X_train, y_train = X[train_idx], y[train_idx]  # make it once and reuse

    # Train one independent neural network
    net, _ = train_classifier(
        train_cfg,
        layers,
        activation_name,
        float(gamma),
        reg_name,
        float(lmbda),
        X_train,
        y_train,
    )

    # Calculate accuracy
    train_acc = accuracy(y_train, predict_labels(net, X_train))

    val_acc = accuracy(y[val_idx], predict_labels(net, X[val_idx]))

    return train_acc, val_acc


def run_mnist(cfg):
    """
    Perform stratified k-fold cross-validation for MNIST classification.

    Evaluates combinations of learning rates and regularization strengths
    while keeping the network architecture, activation function,
    optimizer, and regularization method fixed.

    Selects the best configuration and applies the one-standard-error
    rule based on cross-validation accuracy.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing MNIST, classification,
        cross-validation, and parallelization settings.

    Returns
    -------
    None
        Prints results and saves training and cross-validation
        accuracies, standard errors, and selected parameters.
    """
    c = cfg["mnist_cv"]
    cl = cfg["classification"]

    if c["hyperparameters"] == "load":
        layers, activation_name, reg_name, optimizer_name = load_mnist_settings(cfg)
    else:
        layers, activation_name, reg_name, optimizer_name = selected_mnist_settings(cfg)

    train_cfg = {
        **cfg,
        "classification": {
            **cl,
            "optimizer": optimizer_name,
        },
    }

    # Only the official MNIST training set is used; no test-set evaluation here.
    X, y, _, _ = load_mnist(cl["data_path"])

    n_train = c.get("n_train", len(X))

    # Select an equally balanced subset of MNIST
    if n_train < len(X):
        rng = np.random.default_rng(cfg["data"]["seed"])

        n_classes = len(np.unique(y))
        n_per_class = n_train // n_classes

        indices = np.concatenate(
            [
                rng.choice(np.where(y == digit)[0], size=n_per_class, replace=False)
                for digit in np.unique(y)
            ]
        )

        # Shuffle the selected samples
        rng.shuffle(indices)

        X = X[indices]
        y = y[indices]

    gammas = np.asarray(c["learning_rates"], dtype=float)
    lambdas = np.asarray(c["lambdas"], dtype=float)
    n_splits = cfg["cross_validation"]["n_folds"]
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=cfg["data"]["seed"]
    )
    folds = list(cv.split(X, y))
    shape = (len(gammas), len(lambdas))
    cv_acc = np.empty(shape)
    train_acc = np.empty(shape)
    cv_se = np.empty(shape)

    print(
        f"MNIST CV: layers={layers}, activation={activation_name}, "
        f"regularizer={reg_name}, optimizer={optimizer_name}"
    )
    print(
        f"Training images: {len(X)}, folds={n_splits}, "
        f"combinations={gammas.size * lambdas.size}"
    )

    # Create all independent CV tasks
    tasks = [
        (i, j, train_idx, val_idx)
        for i in range(len(gammas))
        for j in range(len(lambdas))
        for train_idx, val_idx in folds
    ]

    # Parallel execution
    results = Parallel(n_jobs=cfg["parallel"]["n_jobs"], backend="loky", verbose=10)(
        delayed(train_cv_fold)(
            train_cfg,
            layers,
            activation_name,
            reg_name,
            gammas[i],
            lambdas[j],
            X,
            y,
            train_idx,
            val_idx,
        )
        for i, j, train_idx, val_idx in tasks
    )

    # Reconstruct accuracy matrices
    for i in range(len(gammas)):
        for j in range(len(lambdas)):

            indices = [
                n for n, task in enumerate(tasks) if task[0] == i and task[1] == j
            ]

            fold_train = [results[n][0] for n in indices]
            fold_val = [results[n][1] for n in indices]

            train_acc[i, j] = np.mean(fold_train)
            cv_acc[i, j] = np.mean(fold_val)

            cv_se[i, j] = np.std(fold_val, ddof=1) / np.sqrt(n_splits)

            print(
                f"gamma={gammas[i]:.3g} "
                f"lambda={lambdas[j]:.3g}: "
                f"train={train_acc[i,j]:.4f}, "
                f"CV={cv_acc[i,j]:.4f} +/- {cv_se[i,j]:.4f}"
            )

    # For accuracy, larger is better. The 1-SE threshold is *below* the maximum.
    best = np.unravel_index(np.argmax(cv_acc), shape)
    threshold = cv_acc[best] - cv_se[best]
    within = cv_acc >= threshold
    candidates = np.argwhere(within)
    one_se = tuple(max(candidates, key=lambda ij: (lambdas[ij[1]], cv_acc[tuple(ij)])))

    print(
        f"Best CV accuracy: {cv_acc[best]:.4f} +/- {cv_se[best]:.4f} "
        f"(gamma={gammas[best[0]]:.3g}, lambda={lambdas[best[1]]:.3g})"
    )
    print(
        f"1-SE largest lambda: gamma={gammas[one_se[0]]:.3g}, "
        f"lambda={lambdas[one_se[1]]:.3g}, accuracy={cv_acc[one_se]:.4f}"
    )

    _, result_dir = make_output_dirs("cross_validation_mnist")
    save_results(
        {
            "gammas": gammas,
            "lambdas": lambdas,
            "train_acc": train_acc,
            "cv_acc": cv_acc,
            "cv_se": cv_se,
            "best_index": np.asarray(best),
            "one_se_index": np.asarray(one_se),
            "within": within,
            "architecture": np.asarray(layers),
            "activation": np.asarray(activation_name),
            "regularizer": np.asarray(reg_name),
            "optimizer": np.asarray(optimizer_name),
        },
        result_dir / "mnist_cv.npz",
    )


def plot_mnist(cfg=None):
    """
    Plot training and cross-validation accuracy for MNIST.

    Creates two heatmaps showing classification accuracy for different
    learning rates and regularization strengths. Marks the highest
    accuracy and the one-standard-error selection.

    Parameters
    ----------
    cfg : dict or None, optional
        Configuration dictionary. Currently unused.

    Returns
    -------
    None
        Saves the figure as a PDF.
    """
    fig_dir, result_dir = make_output_dirs("cross_validation_mnist")
    res = load_results(result_dir / "mnist_cv.npz")
    gammas, lambdas = res["gammas"], res["lambdas"]
    train_acc, cv_acc = res["train_acc"], res["cv_acc"]
    best = tuple(res["best_index"].astype(int))
    one_se = tuple(res["one_se_index"].astype(int))

    fig, axes = plt.subplots(1, 2, figsize=(7.06, 3.0))
    vmin = min(train_acc.min(), cv_acc.min())
    vmax = max(train_acc.max(), cv_acc.max())

    for i, (ax, values, title) in enumerate(
        zip(
            axes,
            [train_acc, cv_acc],
            ["Training accuracy", "Cross-validation accuracy"],
        )
    ):
        sns.heatmap(
            values.T,
            annot=True,
            fmt=".4f",
            cmap=HEATMAP_CMAP,
            annot_kws={"fontsize": 8},
            vmin=vmin,
            vmax=vmax,
            xticklabels=[scientific_label(g) for g in gammas],
            yticklabels=[scientific_label(l) for l in lambdas],
            cbar=(i == 1),  # only first plot gets colorbar
            cbar_kws={"label": "Accuracy"},
            ax=ax,
        )
        mark_zeros(ax, values.T)
        ax.set_xlabel(r"Learning rate $\gamma$")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)
        ax.set_title(title)
        # Best training accuracy
        train_best = np.unravel_index(np.nanargmax(train_acc), train_acc.shape)

        highlight_cell(axes[0], *train_best)
        highlight_cell(axes[1], *best)
        highlight_cell(axes[1], *one_se, color="C2", linestyle="--")

        if i == 0:
            ax.set_ylabel(r"Regularization $\lambda$")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", left=False, labelleft=False)

    for i, j in np.argwhere(res["within"]):
        # black star with white edge, visible on both light and dark cells
        axes[1].plot(
            i + 0.85, j + 0.22, marker="*", markersize=7,
            markerfacecolor="black", markeredgecolor="white", markeredgewidth=0.5,
        )

    # fig.suptitle(f"MNIST CV: {list(res['architecture'])}, "
    # f"{str(res['activation'])}, {str(res['regularizer'])}, {str(res['optimizer'])}")
    save_figure(fig, "mnist_cv_heatmap.pdf", fig_dir=fig_dir)


def run(cfg):
    """
    Run the selected cross-validation experiment.

    Selects either Runge regression or MNIST classification
    based on the dataset specified in the configuration.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing the dataset selection
        and experiment settings.

    Returns
    -------
    None
        Runs the selected experiment and saves its results.
    """
    if cfg["cross_validation"].get("dataset", "runge") == "mnist":
        return run_mnist(cfg)
    return run_runge(cfg)


def plot(cfg=None):
    """
    Plot the results of the selected cross-validation experiment.

    Selects the Runge or MNIST plotting function based on the
    dataset specified in the configuration.

    Parameters
    ----------
    cfg : dict or None, optional
        Configuration dictionary containing the dataset selection.
        Defaults to Runge regression when None.

    Returns
    -------
    None
        Saves the selected cross-validation figure as a PDF.
    """
    if cfg is not None and cfg["cross_validation"].get("dataset", "runge") == "mnist":
        return plot_mnist(cfg)
    return plot_runge(cfg)
