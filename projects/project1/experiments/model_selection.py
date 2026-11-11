# experiments/model_selection.py

# ----------------------------------------------------------------------
# Task i: Final model selection with cross-validation
# 1. OLS: cross-validation MSE over polynomial degree
# 2. Ridge and Lasso: cross-validation MSE over polynomial degree and lambda.
#    Lasso with our own gradient descent, and with Scikit-Learn as a reference
#    (separates the model from the optimizer)
# 3. Best model for each method according to the cross-validated test error
# 4. Bias-variance decomposition (bootstrap, part c) at common degrees, to see
#    which term each method reduces
#
# Data: the [cv] data set is split once into a training part and a test part
# (same seed and test_size as the bootstrap). Cross-validation (KFold, fixed seed,
# scaling fitted on the training folds in Model.fit) and all hyperparameter choices
# use the training part only; the test part is used once for the final evaluation.
# ----------------------------------------------------------------------

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from .common import sklearn_lasso_fit, logspace
from .cross_validation import (
    get_cv_split,
)  # splits as opposed to get_cv_data which returns the full dataset
from models import RegressionEstimator
from metrics import mse
from data import design_matrix, split_data, runge_function
from resampling import cross_validation_sklearn, bias_variance_bootstrap
import plotting as plot

# ----------------------------------------------------------------------
#   Helpers
# ----------------------------------------------------------------------


class SklearnLasso(BaseEstimator, RegressorMixin):
    """
    Scikit-Learn's Lasso with our scaling and lambda convention, in the same form as
    RegressionEstimator in models.py, so it can be cross-validated in exactly the
    same way. Used as a reference for our own gradient-descent Lasso.

    Parameters
    ----------
    lmbda : float           Lasso regularization parameter (our convention).

    Attributes
    ----------
    model_ : Model          Fitted model, created by fit.
    """

    def __init__(self, lmbda=1.0):
        self.lmbda = lmbda

    def fit(self, X, y):
        """
        Fit Scikit-Learn's Lasso on the training data (scaled inside sklearn_lasso_fit).

        Parameters
        ----------
        X : np.ndarray          Training design matrix.
        y : np.ndarray          Training target values.

        Returns
        -------
        self : SklearnLasso     The fitted estimator.
        """
        self.model_ = sklearn_lasso_fit(X, y, self.lmbda)
        return self

    def predict(self, X):
        """
        Predict target values with the fitted model.

        Parameters
        ----------
        X : np.ndarray          Design matrix.

        Returns
        -------
        np.ndarray              Predicted target values.
        """
        return self.model_.predict(X)


def make_estimator(method, lmbda, cfg):
    """
    Create a Scikit-Learn compatible estimator for one method and lambda.

    Parameters
    ----------
    method : str            "ols", "ridge" (closed form), "lasso" (own gradient descent)
                            or "lasso_sklearn" (Scikit-Learn reference).
    lmbda : float or None   Regularization parameter (ignored for OLS).
    cfg : dict              Configuration dictionary (gradient-descent settings for
                            Lasso are taken from [lasso]).

    Returns
    -------
    estimator               RegressionEstimator or SklearnLasso instance.
    """
    if method == "lasso_sklearn":
        return SklearnLasso(lmbda)

    if method == "lasso":
        lasso_cfg = cfg["lasso"]
        gd_settings = {
            "max_iter": lasso_cfg["max_iter"],
            "tol": lasso_cfg["tol"],
            "gamma_factor": lasso_cfg["gamma_factor"],  # fraction of gamma_max
        }
        return RegressionEstimator(
            method="lasso", lmbda=lmbda, optimizer_kwargs=gd_settings
        )

    return RegressionEstimator(
        method=method, lmbda=lmbda, optimizer_kwargs=None
    )  # closed form OLS and Ridge


# ----------------------------------------------------------------------
#   1. and 2. Cross-validation over degree and lambda
# ----------------------------------------------------------------------


def cv_over_deg_and_lambda(cfg, x, y, method, degrees, lambdas=None):
    """
    k-fold cross-validation MSE on a grid of polynomial degrees and lambda values
    for one method. OLS has no lambda: with lambdas=None the grid has one column.

    Parameters
    ----------
    cfg : dict                  Configuration dictionary.
    x : array-like              Input features.
    y : array-like              Target values.
    method : str                "ols", "ridge", "lasso" or "lasso_sklearn".
    degrees : array-like        Polynomial degrees.
    lambdas : array-like        Regularization parameters; None for OLS.

    Returns
    -------
    mse : np.ndarray            Mean CV MSE, shape (n_degrees, n_lambdas);
                                n_lambdas = 1 for OLS. NaN where a run diverged.
    std : np.ndarray            Standard deviation of the MSE over the folds, same shape.
    """
    ms_cfg = cfg["model_selection"]

    if lambdas is None:
        lambdas = [None]  # OLS: one column, lambda is ignored

    shape = (len(degrees), len(lambdas))
    mse = np.zeros(shape)
    std = np.zeros(shape)

    for i, d in enumerate(degrees):
        X = design_matrix(x, d)  # make design matrix for each degree
        for j, lmbda in enumerate(lambdas):
            estimator = make_estimator(method, lmbda, cfg)
            with np.errstate(over="ignore", invalid="ignore"):  # diverging GD runs
                mse[i, j], std[i, j] = cross_validation_sklearn(
                    X,
                    y,
                    k=ms_cfg["k"],
                    method=method,
                    lmbda=lmbda,
                    random_state=cfg["data"]["seed"],
                    estimator=estimator,
                )

    mse[~np.isfinite(mse)] = (
        np.nan
    )  # diverged runs are skipped when finding the best model
    return mse, std


# ----------------------------------------------------------------------
#   3. Best model per method
# ----------------------------------------------------------------------


def best_lambda_deg(degrees, lambdas, mse, std):
    """
    Find the degree and lambda with the lowest cross-validated MSE.
    Works for OLS as well, with lambdas=[None].

    Parameters
    ----------
    degrees : array-like        Polynomial degrees (rows of mse).
    lambdas : array-like        Lambda values (columns of mse); [None] for OLS.
    mse : np.ndarray            Mean CV MSE, shape (n_degrees, n_lambdas).
    std : np.ndarray            Standard deviation over the folds, same shape.

    Returns
    -------
    degree : int                Best polynomial degree.
    lmbda : float or None       Best lambda (None for OLS).
    mse : float                 CV MSE of the best model.
    std : float                 Its standard deviation over the folds.
    """
    i, j = np.unravel_index(np.nanargmin(mse), mse.shape)
    return degrees[i], lambdas[j], mse[i, j], std[i, j]


def heldout_mse(cfg, method, degree, lmbda, x_train, y_train, x_test, y_test):
    """Refit on the full training part at the chosen (degree, lambda) and compute MSE on the test part."""
    estimator = make_estimator(method, lmbda, cfg)
    with np.errstate(over="ignore", invalid="ignore"):
        estimator.fit(design_matrix(x_train, degree), y_train)
        y_pred = estimator.predict(design_matrix(x_test, degree))
    return mse(y_test, y_pred)


# ----------------------------------------------------------------------
#   4. Bias-variance decomposition
# ----------------------------------------------------------------------


def bias_variance_for(
    cfg: dict, method: str, degree: int, lmbda, x_train, y_train, x_test, y_test
):
    """
    Bootstrap estimate of bias^2 (relative to the true Runge function), variance and
    test MSE for one method at one degree and lambda, with the setup from part c.
    The same seed is used for every method, so all methods see the same bootstrap samples.

    Parameters
    ----------
    cfg : dict                  Configuration dictionary.
    method : str                "ols", "ridge" or "lasso" (own gradient descent).
    degree : int                Polynomial degree.
    lmbda : float or None       Regularization parameter (None for OLS).
    x_train : array-like        Training input features.
    y_train : array-like        Training target values.
    x_test : array-like         Test input features.
    y_test : array-like         Test target values.

    Returns
    -------
    bias2 : float               Squared bias relative to the true function.
    variance : float            Variance of the predictions over the bootstrap models.
    mse : float                 Test MSE averaged over the bootstrap models.
    """

    if method == "lasso":
        lasso_cfg = cfg["lasso"]
        lasso_kwargs = {
            "gamma_factor": lasso_cfg["gamma_factor"],
            "max_iter": lasso_cfg["max_iter"],
            "tol": lasso_cfg["tol"],
        }
    else:
        lasso_kwargs = {}

    seed = cfg["data"]["seed"]
    np.random.seed(seed)  # same bootstrap samples for every method

    with np.errstate(
        over="ignore", invalid="ignore"
    ):  # errstate to ignore diverging runs error messages
        return bias_variance_bootstrap(
            design_matrix(x_train, degree),
            y_train,
            design_matrix(x_test, degree),
            y_test,
            f_test=runge_function(x_test),
            method=method,
            lmbda=lmbda,
            n_bootstrap=cfg["model_selection"]["n_bootstrap"],
            bias_reference="f",
            lasso_kwargs=lasso_kwargs,
        )


# ----------------------------------------------------------------------
#   Run everything
# ----------------------------------------------------------------------


def run_model_selection(cfg: dict) -> None:
    """
    Run the final model selection for task i:

    1. Cross-validated MSE for OLS over degree, and for Ridge and Lasso (own GD and
       Scikit-Learn) over degree and lambda.
    2. The best model for each method (printed table).
    3. Figures: CV MSE vs degree for all methods (best lambda per degree), and
       heatmaps of the Ridge and Lasso grids.
    4. Bootstrap bias-variance decomposition at the degrees in
       [model_selection] bias_variance_degrees (printed table).

    Parameters
    ----------
    cfg : dict      Configuration dictionary containing the [model_selection], [cv],
                    [lasso] and [data] settings.
    """
    ms_cfg = cfg["model_selection"]
    x_train, x_test, y_train, y_test = get_cv_split(cfg)
    k = ms_cfg["k"]
    degrees = np.arange(1, ms_cfg["max_degree"] + 1)
    lasso_degrees = np.arange(1, ms_cfg["lasso_max_degree"] + 1)
    ridge_lambdas = logspace(ms_cfg["ridge_lambda_log"])
    lasso_lambdas = logspace(ms_cfg["lasso_lambda_log"])

    # 1. and 2. cross-validation
    mse_ols, std_ols = cv_over_deg_and_lambda(
        cfg, x_train, y_train, "ols", degrees, lambdas=None
    )
    mse_ridge, std_ridge = cv_over_deg_and_lambda(
        cfg, x_train, y_train, "ridge", degrees, ridge_lambdas
    )
    mse_lasso, std_lasso = cv_over_deg_and_lambda(
        cfg, x_train, y_train, "lasso", lasso_degrees, lasso_lambdas
    )
    mse_lasso_sk, std_lasso_sk = cv_over_deg_and_lambda(
        cfg, x_train, y_train, "lasso_sklearn", lasso_degrees, lasso_lambdas
    )

    # 3. best model per method
    best = {
        "OLS": best_lambda_deg(degrees, [None], mse_ols, std_ols),
        "Ridge": best_lambda_deg(degrees, ridge_lambdas, mse_ridge, std_ridge),
        "Lasso (own GD)": best_lambda_deg(
            lasso_degrees, lasso_lambdas, mse_lasso, std_lasso
        ),
        "Lasso (Sk-Learn)": best_lambda_deg(
            lasso_degrees, lasso_lambdas, mse_lasso_sk, std_lasso_sk
        ),
    }

    method_key = {
        "OLS": "ols",
        "Ridge": "ridge",
        "Lasso (own GD)": "lasso",
        "Lasso (Sk-Learn)": "lasso_sklearn",
    }

    print(
        f"\nFinal model selection, {k}-fold cross-validation on the training part "
        f"(n_train = {len(x_train)}, n_test = {len(x_test)}, "
        f"noise variance {cfg['data']['sigma'] ** 2:.3g})"
    )
    print("-" * 78)
    print(
        f"{'Method':<22} {'degree':>7} {'lambda':>10} {'CV MSE':>10} "
        f"{'std (folds)':>12} {'test MSE':>10}"
    )
    for name, (degree, lmbda, cv_mse, std) in best.items():
        t_mse = heldout_mse(
            cfg, method_key[name], degree, lmbda, x_train, y_train, x_test, y_test
        )
        lam = "-" if lmbda is None else f"{lmbda:.0e}"
        print(
            f"{name:<22} {degree:>7} {lam:>10} {cv_mse:10.5f} {std:12.5f} {t_mse:10.5f}"
        )
    winner = min(best, key=lambda name: best[name][2])  # chosen by CV MSE, not test MSE
    print(f"Lowest CV MSE: {winner}")

    # figures: CV MSE vs degree (best lambda per degree), and the two grids
    curves = [
        (degrees, np.nanmin(mse_ols, axis=1)),
        (degrees, np.nanmin(mse_ridge, axis=1)),
        (lasso_degrees, np.nanmin(mse_lasso, axis=1)),
        (lasso_degrees, np.nanmin(mse_lasso_sk, axis=1)),
    ]
    labels = ["OLS", "Ridge", "Lasso (own)", "Lasso (Sk-Learn)"]

    plot.plot_cv_mse_methods(curves, labels, k, styles=["-", "-", "-", "--"])
    plot.plot_cv_heatmap(degrees, ridge_lambdas, mse_ridge, "ridge", k)
    plot.plot_cv_heatmap(lasso_degrees, lasso_lambdas, mse_lasso, "lasso", k)

    # 4. bias-variance at common degrees (each method with its best lambda at that degree)
    print(
        f"\nBias-variance decomposition (bootstrap, {ms_cfg['n_bootstrap']} samples, "
        f"bias relative to the true function)"
    )
    print(
        "-----------------------------------------------------------------------------"
    )
    print(
        f"{'Method':<22} {'degree':>7} {'lambda':>10} {'bias^2':>10} {'variance':>10} {'test MSE':>10}"
    )

    for d in ms_cfg["bias_variance_degrees"]:
        j_ridge = np.nanargmin(mse_ridge[list(degrees).index(d)])
        rows = [
            ("OLS", "ols", None),
            ("Ridge", "ridge", ridge_lambdas[j_ridge]),
        ]
        if d in lasso_degrees:  # Lasso only for degrees in its grid
            j_lasso = np.nanargmin(mse_lasso[list(lasso_degrees).index(d)])
            rows.append(("Lasso (own GD)", "lasso", lasso_lambdas[j_lasso]))

        for name, method, lmbda in rows:
            bias2, variance, mse_bv = bias_variance_for(
                cfg, method, d, lmbda, x_train, y_train, x_test, y_test
            )
            lam = "-" if lmbda is None else f"{lmbda:.0e}"
            print(
                f"{name:<22} {d:>7} {lam:>10} {bias2:10.5f} {variance:10.5f} {mse_bv:10.5f}"
            )
