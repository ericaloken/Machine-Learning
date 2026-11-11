# regularization.py
"""
Part d: L1 and L2 regularization of feed-forward neural networks on the
noisy Runge function, compared with Ridge and Lasso from Project 1.

Main functions
--------------
run()   Train the networks, evaluate Ridge and Lasso, save results to
        regularization_study.npz and generate the figures by calling plot().

plot()  Load regularization_study.npz and regenerate the figures without
        retraining.


Pipeline
--------
1. Data   load_data()

   Generate noisy Runge data using the same train/validation/test split
   as in the previous tasks.

   Train: fit the network weights.
   Val: choose the architecture, learning rate and regularization strength.
   Test: evaluate the selected models.


2. Regularization study   run_regularization_study()

   For each regularizer (L1 and L2):
     - train networks with different architectures, learning rates (gamma)
       and regularization strengths (lambda)
     - train the lambda-gamma combinations in parallel for each architecture
     - select the configuration with the lowest validation MSE
     - store only the selected model's training history
     - evaluate the selected network on the test set
     - predict on a dense grid for plotting the regression fit


3. Polynomial benchmarks   ridge_benchmark(), lasso_benchmark()

   Fit Ridge and Lasso using the polynomial degrees and regularization
   strengths selected in Project 1.

   Both models are fitted on the training set and evaluated on the same
   test set as the neural networks.

   Predictions on a dense grid are also calculated for comparison.


4. Save and plot   save_results(), print_summary(), plot()

   Save the results, print the selected hyperparameters and test MSE,
   and generate the regularization heatmaps and regression-fit figure.


Saved results (output/regularization_study/results/regularization_study.npz)
---------------------------------------------------------------------------
Settings:  regularizers, architectures, learning_rates, lambdas

Test set:  x_test, y_test

Plot grid: x_grid

Neural networks, with:
    R = n_regularizers
    A = n_architectures
    L = n_lambdas
    G = n_learning_rates

    train_mse       (R, A, L, G)    training MSE of every run
    val_mse         (R, A, L, G)    validation MSE of every run
    best_arch       (R,)            selected architecture index
    best_lambda     (R,)            selected regularization strength
    best_gamma      (R,)            selected learning rate
    history         (R, epochs)     training loss of selected models
    test_mse        (R,)            test MSE of selected models
    test_pred       (R, n_test)     test predictions of selected models
    grid_pred       (R, n_grid)     grid predictions of selected models

Polynomial benchmarks:
    ridge_test_mse                  Ridge test MSE
    ridge_test_pred                 Ridge test predictions
    ridge_grid_pred                 Ridge grid predictions
    lasso_test_mse                  Lasso test MSE
    lasso_test_pred                 Lasso test predictions
    lasso_grid_pred                 Lasso grid predictions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from joblib import Parallel, delayed

from regularizers import L1, L2
from experiments.classification import REGULARIZER_LABELS
from activations import Sigmoid, ReLU
from linear_models import ridge_fit, sklearn_lasso_fit
from data import design_matrix, runge_function
from experiments.regression import load_data
from metrics import mse
from plotting_utils import save_figure, scientific_label, legend_below, HEATMAP_CMAP, mark_zeros, highlight_cell
from experiments.common import (
    train_network,
    predict,
    make_output_dirs,
    save_results,
    load_results,
)

TASK = "regularization_study"
RESULTS_NAME = "regularization_study.npz"

REGULARIZERS = {
    "l1": L1,
    "l2": L2,
}

ACTIVATIONS = {
    "sigmoid": Sigmoid,
    "relu": ReLU,
}


# ------------------------------------------------------------
# Ridge and Lasso benchmarks
# ------------------------------------------------------------

def ridge_benchmark(cfg, data):
    """
    Ridge with best polynomial degree and lambda from project1.

    Fitted on train + val, tested on test set.
    Model scales X, y with training statistics and predict() returns
    values on original scale, as in p1.

    Returns
    -------
    dict:
        "model" : fitted model
        "test_mse" : MSE on test set
        "grid_pred" : predictions on grid for plotting
    """
    degree, lmbda = cfg["regularization_study"]["best_ridge"]

    x_train, y_train = data["train"]
    x_test, y_test = data["test"]

    model = ridge_fit(design_matrix(x_train, degree), y_train, lmbda)
    test_pred = model.predict(design_matrix(x_test, degree))

    # grid for smooth plotting
    x_grid = np.linspace(-1, 1, 500)
    grid_pred = model.predict(design_matrix(x_grid, degree))

    return test_pred, mse(y_test, test_pred), grid_pred


def lasso_benchmark(cfg, data):
    """
    Lasso with best polynomial degree and lambda from project1.

    Fitted on train + val, tested on test set.
    Model scales X, y with training statistics and predict() returns
    values on original scale, as in p1.

    Returns
    -------
    dict:
        "model" : fitted model
        "test_mse" : MSE on test set
        "grid_pred" : predictions on grid for plotting
    """
    degree, lmbda = cfg["regularization_study"]["best_lasso"]
    x_train, y_train = data["train"]
    x_test, y_test = data["test"]

    model = sklearn_lasso_fit(design_matrix(x_train, degree), y_train, lmbda)

    test_pred = model.predict(design_matrix(x_test, degree))

    # grid for smooth plotting
    x_grid = np.linspace(-1, 1, 500)
    grid_pred = model.predict(design_matrix(x_grid, degree))

    return test_pred, mse(y_test, test_pred), grid_pred


# ------------------------------------------
# Regularization and architecture study
# ------------------------------------------


def train_and_evaluate(cfg, layers, regularizer_name, lmbda, gamma, data):
    """
    Train a regularized network and evaluate the training and validation MSE.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.
    layers : list[int]
        Network architecture, including input and output layers.
    regularizer_name : str
        Type of regularization, either "l1" or "l2".
    lmbda : float
        Regularization strength.
    gamma : float
        Learning rate.
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
    study = cfg["regularization_study"]

    x_train, y_train = data["train"]
    x_val, y_val = data["val"]

    regularizer = REGULARIZERS[regularizer_name](lmbda)

    net, scaler, history = train_network(
        cfg,
        layers,
        optimizer_name=study["optimizer"],
        gamma=gamma,
        x_train=x_train,
        y_train=y_train,
        hidden_activation=ACTIVATIONS[study["activation"]],
        regularizer=regularizer
    )
    train_mse = mse(y_train, predict(net, scaler, x_train))
    val_mse = mse(y_val, predict(net, scaler, x_val))

    return net, scaler, train_mse, val_mse, history


def run_regularization_study(cfg):
    """
    Study L1 and L2 regularization for different architectures,
    learning rates and regularization strengths.

    Select the best configuration for each regularizer using
    validation MSE, then evaluate it on the test set.

    Returns
    -------
    dict
        Training and validation MSE, selected hyperparameters,
        test MSE and predictions for each regularizer.
    """
    study = cfg["regularization_study"]
    n_jobs = cfg["parallel"]["n_jobs"]

    data = load_data(cfg, n_points=study["n_points"])
    x_test, y_test = data["test"]

    # holding grid values for plotting the fit
    x_grid = np.linspace(-1, 1, 500)
    grid_pred = np.full((len(study["regularizers"]), len(x_grid)), np.nan)

    regularizers = study["regularizers"]
    architectures = study["architectures"]
    gammas = study["learning_rates"]
    lambdas = study["lambdas"]
    epochs = cfg["training"]["epochs"]

    # initialize with nan so that diverged runs are skipped
    shape = (
        len(regularizers),
        len(architectures),
        len(lambdas),
        len(gammas),
    )
    val_mse = np.full(shape, np.nan)
    train_mse = np.full(shape, np.nan)
    history = np.full((len(regularizers), epochs), np.nan)
    best_arch = np.full(len(regularizers), -1, dtype=int)
    best_gamma = np.full(len(regularizers), np.nan)
    best_lambda = np.full(len(regularizers), np.nan)

    # store test mse for the selected architecture for each activation
    test_mse = np.full(len(regularizers), np.nan)
    test_pred = np.full((len(regularizers), len(y_test)), np.nan)

    for r, regularizer_name in enumerate(regularizers):
        best_val, best_model = np.inf, None

        for i, layers in enumerate(architectures): # test differenet architectures
            # create all combinations of gamma and lambda for paralelization
            hyperparam_combos = [
                (l, g, lmbda, gamma)
                for l, lmbda in enumerate(lambdas)
                for g, gamma in enumerate(gammas)
                ]

            # train each combination in parallel
            runs = Parallel(n_jobs=n_jobs)(
                delayed(train_and_evaluate)(
                    cfg,
                    layers,
                    regularizer_name,
                    lmbda,
                    gamma, 
                    data,
                )
                for l, g, lmbda, gamma in hyperparam_combos
            )
            # recover results correspoding to each combination
            for (l, g, lmbda, gamma), result in zip(hyperparam_combos, runs):
                net, scaler, train, val, hist = result

                train_mse[r, i, l, g] = train
                val_mse[r, i, l, g] = val

                print(
                    f"{regularizer_name:>3} "
                    f"{str(layers):>10} "
                    f"lambda={lmbda:.0e}, "
                    f"gamma={gamma:.0e}: "
                    f"train={train:.5f}, val={val:.5f}"
                )
                # select model with lowest validation MSE
                if np.isfinite(val) and val < best_val:  # avoids NaN issues
                    best_val, best_model = val, (net, scaler)
                    best_arch[r] = i
                    best_lambda[r] = lmbda
                    best_gamma[r] = gamma
                    history[r, :] = np.nan
                    history[r, :len(hist)] = hist

        # evaluate best model on test data
        if best_model is not None:
            test_pred[r] = predict(*best_model, x_test)
            test_mse[r] = mse(y_test, test_pred[r])
            grid_pred[r] = predict(*best_model, x_grid)

    return {
        "regularizers": np.array(regularizers),
        "architectures": np.array(
            [str(layers[1:-1]) for layers in architectures]
        ),
        "learning_rates": np.array(gammas),
        "lambdas": np.array(lambdas),
        "train_mse": train_mse,
        "val_mse": val_mse,
        "history": history,
        "best_arch": best_arch,
        "best_lambda": best_lambda,
        "best_gamma": best_gamma,
        "test_mse": test_mse,
        "test_pred": test_pred,
        "x_test": x_test,
        "y_test": y_test,
        "x_grid": x_grid,
        "grid_pred": grid_pred,
    }



# -------------------------------------------------------------
# Plotting:
# plotting functions made by ChatGPT-6 based on the
# docstrings and detailed prompts.
# -------------------------------------------------------------


def print_summary(results) -> None:
    """
    Print the validation-selected configuration for L1 and L2,
    together with its test MSE.
    """
    print("\n" + "=" * 85)
    print("TASK D: REGULARIZATION STUDY")
    print("=" * 85)

    print(
        f"{'Penalty':<10}"
        f"{'Architecture':<18}"
        f"{'Lambda':>12}"
        f"{'Gamma':>12}"
        f"{'Val MSE':>15}"
        f"{'Test MSE':>15}"
    )
    print("-" * 85)

    for r, name in enumerate(results["regularizers"]):
        i = int(results["best_arch"][r])

        if i < 0:
            continue

        best_val = np.nanmin(results["val_mse"][r])

        print(
            f"{name:<10}"
            f"{results['architectures'][i]:<18}"
            f"{results['best_lambda'][r]:>12.1e}"
            f"{results['best_gamma'][r]:>12.1e}"
            f"{best_val:>15.5f}"
            f"{results['test_mse'][r]:>15.5f}"
        )
    print("=" * 85)
    print("\nPOLYNOMIAL REGRESSION BENCHMARKS")
    print("-" * 40)
    print(f"Ridge test MSE: {results['ridge_test_mse']:.5f}")
    print(f"Lasso test MSE: {results['lasso_test_mse']:.5f}")




def plot_regularization_heatmaps(results, fig_dir) -> None:
    """
    Plot validation MSE heatmaps for regularization study.

    For each regularizer (L1 and L2), make two figures:
    1) architectures [50] and [50, 50]
    2) architectures [100] and [100, 100]

    The lowest validation MSE in each heatmap is highlighted.
    """
    architectures = list(results["architectures"])
    gammas = results["learning_rates"]
    lambdas = results["lambdas"]

    # Group architectures into two figures
    groups = {
        "50s": [i for i, a in enumerate(architectures) if a in ("[50]", "[50, 50]")],
        "100s": [i for i, a in enumerate(architectures) if a in ("[100]", "[100, 100]")],
    }

    for r, reg_name in enumerate(results["regularizers"]):
        all_values = results["val_mse"][r]

        for group_name, idxs in groups.items():
            if len(idxs) == 0:
                continue

            # values for the selected architectures in this figure
            group_values = all_values[idxs]

            fig, axes = plt.subplots(
                1, len(idxs),
                figsize=(7.06, 2.8 if len(idxs) == 2 else 3.0),
                sharey=True,
                layout="constrained",
            )

            # If there is only one panel, make axes iterable
            axes = np.atleast_1d(axes)

            # Shared color scale within this figure
            vmin = np.nanmin(group_values)
            vmax = np.nanmax(group_values)

            # Shared colorbar, log scale so a few large MSE values do not wash out the rest
            norm = LogNorm(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(norm=norm, cmap=HEATMAP_CMAP)
            sm.set_array([])

            for ax, idx in zip(axes, idxs):
                values = all_values[idx]

                sns.heatmap(
                    values,
                    ax=ax,
                    cmap=HEATMAP_CMAP,
                    norm=norm,
                    annot=True,
                    fmt=".3f",
                    annot_kws={"fontsize": 8},
                    xticklabels=[scientific_label(g) for g in gammas],
                    yticklabels=[scientific_label(lmbda) for lmbda in lambdas],
                    cbar=False,
                    linewidths=0.2,
                    linecolor="gray",
                )
                mark_zeros(ax, values)
                ax.set_title(f"Hidden layers: {architectures[idx]}")
                ax.set_xlabel(r"Learning rate $\gamma$")
                ax.set_ylabel("")
                ax.tick_params(axis="x", rotation=0)
                ax.tick_params(axis="y", rotation=0)
                if ax is not axes[0]:
                    ax.tick_params(
                        axis="y",
                        which="both",
                        left=False,
                        labelleft=False,
                    )
                # Highlight best (minimum) cell
                best_row, best_col = np.unravel_index(np.nanargmin(values), values.shape)
                highlight_cell(ax, best_col, best_row)
            axes[0].set_ylabel(r"Regularization strength $\lambda$")

            cbar = fig.colorbar(sm, ax=axes, shrink=0.88, pad=0.02)
            cbar.set_label("Validation MSE")

            save_figure(fig, f"regularization_heatmap_{reg_name}_{group_name}.pdf", fig_dir=fig_dir, Tight=False)


def plot_regression_fit(results, fig_dir) -> None:
    """
    Compare the Runge function with predictions from the
    best L1/L2 neural networks and Ridge/Lasso regression.
    """
    fig, ax = plt.subplots()

    x_grid = results["x_grid"]
    x_test = results["x_test"]
    y_test = results["y_test"]

    # True Runge function
    ax.plot(
            x_grid,
            runge_function(x_grid),
            color="black",
            linestyle="--",
            label="Runge function",
            zorder=3,
        )

    ax.scatter(
            x_test,
            y_test,
            s=12,
            color="gray",
            alpha=0.5,
            label="Test data",
            zorder=1,
        )

    for r, name in enumerate(results["regularizers"]):
        ax.plot(x_grid, results["grid_pred"][r], label=f"{REGULARIZER_LABELS[str(name)]}-FFNN")
    
    ax.plot(x_grid, results["ridge_grid_pred"], label="Ridge")
    ax.plot(x_grid, results["lasso_grid_pred"], label="Lasso")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.grid()
    legend_below(ax, ncol=3)

    save_figure(fig, "regularization_regression_fit.pdf", fig_dir=fig_dir)

# -------------------------------------------------------------
# Run and plot Task D
# -------------------------------------------------------------

def run(cfg) -> None:
    """
    Run Task D: save the results and generate figures. We:

    - ...
    """
    _, result_dir = make_output_dirs(TASK)

    # Neural-network experiments
    results = run_regularization_study(cfg)

    # Polynomial regression benchmarks
    data = load_data(
        cfg, n_points=cfg["regularization_study"]["n_points"],
    )

    ridge_pred, ridge_mse, ridge_grid_pred = ridge_benchmark(cfg, data)
    lasso_pred, lasso_mse, lasso_grid_pred = lasso_benchmark(cfg, data)

    results.update({
        "ridge_test_pred": ridge_pred,
        "ridge_test_mse": ridge_mse,
        "ridge_grid_pred": ridge_grid_pred,
        "lasso_test_pred": lasso_pred,
        "lasso_test_mse": lasso_mse,
        "lasso_grid_pred": lasso_grid_pred,
    })

    save_results(results, result_dir / RESULTS_NAME)
    print_summary(results)
    plot(cfg)


def plot(cfg) -> None:
    """
    Plot the saved Task D results.
    """
    fig_dir, result_dir = make_output_dirs(TASK)
    results = load_results(result_dir / RESULTS_NAME)

    plot_regularization_heatmaps(results, fig_dir)
    plot_regression_fit(results, fig_dir)