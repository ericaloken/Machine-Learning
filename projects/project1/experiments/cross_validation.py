# tasks/cross_validation.py

# -------------------------------------------------------------
# Task d: Cross-validation using Scikit-Learn
# - OLS over polynomial degree for k = 5 and k = 10
# - Ridge over polynomial degree for k = 5 and k = 10
# - Ridge over regularization strength for k = 5 and k = 10
# - Compare OLS cross-validation MSE with bootstrap MSE
# - Verify our own k-fold implementation against Scikit-Learn
# --------------------------------------------------------------

from .common import make_model, logspace, get_data
from .bootstrap import run_bias_variance
from data import split_data
from resampling import (
    cross_validation_degree,
    cross_validation_lambda,
    bias_variance_lambda,
)
import plotting as plot


def get_cv_data(cfg: dict):
    """
    Generate the data used for the cross-validation analyses.

    Parameters
    ----------
    cfg : dict              Configuration dictionary containing the cross-validation
                            and data-generation settings.

    Returns
    -------
    x : array-like          Input values.
    y : array-like          Corresponding noisy target values.
    """
    runge = make_model(cfg, n_points=cfg["cv"]["n_points"])
    return get_data(runge)


def get_cv_split(cfg: dict):
    """Train/test split of the CV data."""
    x, y = get_cv_data(cfg)
    return split_data(
        x, y, test_size=cfg["data"]["test_size"], random_state=cfg["data"]["seed"]
    )  # x_train, x_test, y_train, y_test


def run_cv_ols_over_degree(x, y, cfg: dict, degrees, k: int):
    """
    Evaluate OLS using k-fold cross-validation over polynomial degree.

    Parameters
    ----------
    x, y : array-like      Input features and target values.
    cfg : dict              Configuration dictionary containing the cross-validation
                            and data-generation settings.
    degrees : iterable      Polynomial degrees included in the analysis.
    k : int                 Number of cross-validation folds.

    Returns
    -------
    mse : array-like        Mean cross-validation MSE for each polynomial degree.
    std : array-like        Standard deviation of the MSE across the folds.
    """
    mse, std = cross_validation_degree(
        x,
        y,
        degrees=degrees,
        method="ols",
        k=k,
        random_state=cfg["data"]["seed"],
    )
    return mse, std


def run_cv_ridge_over_lambda(x, y, cfg: dict, k: int):
    """
    Evaluate Ridge regression using k-fold cross-validation over lambda,
    with fixed polynomial degree.

    Parameters
    ----------
    x : array-like          Input features.
    y : array-like          Target values.
    cfg : dict                 Configuration dictionary containing the cross-validation
                               and Ridge settings.
    k : int                    Number of cross-validation folds.

    Returns
    -------
    lambda_vals : array-like    Regularization parameters included in the analysis.
    mse : array-like            Mean cross-validation MSE for each lambda.
    std : array-like            Standard deviation of the MSE across the folds.
    """
    cv_cfg = cfg["cv"]
    lambda_vals = logspace(cv_cfg["ridge_lambda_test_log"])
    mse, std = cross_validation_lambda(
        x,
        y,
        degree=cv_cfg["ridge_degree"],
        method="ridge",
        k=k,
        random_state=cfg["data"]["seed"],
        lambdas=lambda_vals,
    )
    return lambda_vals, mse, std


def run_cv_ridge_over_degree(x, y, cfg: dict, degrees, k: int):
    """
    Evaluate Ridge regression using k-fold cross-validation over degree,
    with fixed parameter.

    Parameters
    ----------
    x : array-like          Input features.
    y : array-like          Target values.
    cfg : dict              Configuration dictionary containing the cross-validation
                            and Ridge settings.

    degrees : iterable      Polynomial degrees included in the analysis.
    k : int                 Number of cross-validation folds.

    Returns
    -------
    mse : array-like        Mean cross-validation MSE for each polynomial degree.
    std : array-like        Standard deviation of the MSE across the folds.
    """
    mse, std = cross_validation_degree(
        x,
        y,
        degrees=degrees,
        method="ridge",
        k=k,
        random_state=cfg["data"]["seed"],
        lmbda=cfg["cv"]["ridge_lambda"],
    )
    return mse, std


def run_cv_ols_ridge(cfg: dict):
    """
    Run the OLS and Ridge cross-validation analyses for each k value.
    Three experiments are performed for every number of folds:

    1. OLS performance as a function of polynomial degree.
    2. Ridge performance as a function of lambda at fixed degree.
    3. Ridge performance as a function of polynomial degree at fixed lambda.

    The resulting cross-validation MSE values and fold-to-fold
    standard deviations are plotted.

    Parameters
    ----------
    cfg : dict      Configuration dictionary containing the cross-validation,
                    regression, and plotting settings.
    """
    cv_cfg = cfg["cv"]
    degrees = range(1, cv_cfg["max_degree"] + 1)

    x, y = get_cv_data(cfg)
    x_test, x_train, y_test, y_train = split_data(
        x, y, test_size=0.2, random_state=cfg["data"]["seed"]
    )

    ols_results = {}
    ridge_lambda_results = {}
    ridge_degree_results = {}

    for k in cv_cfg["k_vals"]:
        ols_results[k] = run_cv_ols_over_degree(x, y, cfg, degrees, k)

        ridge_lambda_results[k] = run_cv_ridge_over_lambda(x, y, cfg, k)

        ridge_degree_results[k] = run_cv_ridge_over_degree(
            x,
            y,
            cfg,
            degrees,
            k,
        )

    # Plot k=5 and k=10 together.
    plot.plot_cv_mse_multiple(
        x_values=degrees,
        results=ols_results,
        variable="degree",
        method="ols",
    )

    plot.plot_cv_mse_multiple(
        x_values=ridge_lambda_results[cv_cfg["k_vals"][0]][0],
        results={k: (mse, std) for k, (_, mse, std) in ridge_lambda_results.items()},
        variable="lambda",
        method="ridge",
    )

    plot.plot_cv_mse_multiple(
        x_values=degrees,
        results=ridge_degree_results,
        variable="degree",
        method="ridge",
    )


def run_cv_vs_bootstrap(cfg: dict):
    """
    Compare cross-validation MSE (k = 5 and k = 10) with bootstrap MSE over polynomial
    degree, for OLS and for Ridge at the fixed lambda [cv] ridge_lambda.
    Parameters
    ----------
    cfg : dict      Configuration dictionary containing the cross-validation,
                    bootstrap, and data-generation settings.
    """
    if cfg["cv"]["n_points"] != cfg["bootstrap"]["n_points"]:
        raise ValueError(
            "[cv] and [bootstrap] n_points must match to compare CV and bootstrap."
        )

    x, y = get_cv_data(cfg)
    lmbda = cfg["cv"]["ridge_lambda"]
    for method, lmbda in [("ols", None), ("ridge", lmbda)]:
        degrees, _, _, mse_boot = run_bias_variance(
            cfg, reference="y", linreg_method=method, lmbda=lmbda
        )

        cv_results = {}

        # CV with the same method as the bootstrap
        run_cv_over_degree = (
            run_cv_ridge_over_degree if method == "ridge" else run_cv_ols_over_degree
        )

        for k in cfg["cv"]["k_vals"]:
            mse_cv, std_cv = run_cv_over_degree(
                x,
                y,
                cfg,
                degrees,
                k,
            )

            cv_results[k] = (mse_cv, std_cv)

        plot.plot_cv_bootstrap_mse_comparison(
            degrees,
            mse_boot,
            cv_results,
            method=method,
        )


def run_cv(cfg: dict) -> None:
    """
    Run the complete cross-validation analysis for task d.
    Two analyses are performed:

    1. OLS and Ridge cross-validation for different polynomial degrees
       and regularization parameters.
    2. Comparison of OLS and Ridge cross-validation MSE with bootstrap MSE.

    The verification of our own k-fold implementation against Scikit-Learn
    is a separate task (verify_cv in main.py).

    Parameters
    ----------
    cfg : dict      Configuration dictionary containing the cross-validation,
                    bootstrap, regression, data-generation, and plotting settings.
    """
    run_cv_ols_ridge(cfg)
    run_cv_vs_bootstrap(cfg)
