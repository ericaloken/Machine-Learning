# tasks/ridge.py

# -------------------------------------------------------------------
# Task b: Ridge regression
# - Model performance as a function of the regularization
#   parameter lambda for a fixed polynomial degree
# - Model performance as a function of polynomial degree
#   for a fixed regularization parameter.
# - Plot MSE, R^2, and fitted coefficients for both analyses.
# - Plot the singular values of the scaled design matrix and the corresponding shrinkage factors for the lambda sweep.
# -------------------------------------------------------------------

from .common import make_model, logspace
import numpy as np
from models import Model
import plotting as plot


def singular_values(cfg: dict):
    """
    Singular values of the scaled training design matrix at the degree of the
    lambda sweep (the matrix whose modes Ridge shrinks).

    Returns
    -------
    np.ndarray      Singular values, largest first.
    """
    runge = make_model(cfg)
    X_train, _, y_train, _ = runge.split_design_data(
        degree=cfg["ridge"]["degree"], test_size=cfg["data"]["test_size"]
    )
    mean_X, std_X, _, _ = Model.training_statistics(X_train, y_train)
    X_scaled = Model.scale_X(X_train, mean_X, std_X)
    return np.linalg.svd(X_scaled, compute_uv=False)


def run_ridge_over_lambda(cfg: dict):
    """
    Evaluate Ridge regression over a range of regularization parameters,
    with fixed polynomial degree.

    Parameters
    ----------
    cfg : dict                      Configuration dictionary containing settings and
                                    parameters required to construct the model.
    Returns
    -------
    lambda_vals : array-like        Regularization parameters used in the analysis.
    mse_vals : array-like           MSE values obtained for each lambda.
    r2_vals : array-like            R^2 scores obtained for each lambda.
    parameters : array-like         Fitted Ridge coefficients for each lambda.
    """
    ridge_cfg = cfg["ridge"]
    lambda_test_log = ridge_cfg["lambda_test_log"]
    lambda_vals = logspace(lambda_test_log)
    degree = ridge_cfg["degree"]
    runge = make_model(cfg)
    mse_vals, r2_vals, parameters = runge.evaluate_ridge_multiple_lambdas(
        degree, lambdas=lambda_vals
    )
    return lambda_vals, mse_vals, r2_vals, parameters


def run_ridge_over_degree(cfg: dict):
    """
    Evaluate Ridge regression over polynomial degree for several lambda values,
    with OLS (lambda = 0) as the reference.

    Parameters
    ----------
    cfg : dict                      Configuration dictionary containing settings and
                                    parameters required to construct the model.
    Returns
    -------
    degrees : range                 Polynomial degrees included in the analysis.
    lambda_vals : list               Lambda values used in the degree sweep.
    mse_vals : np.ndarray           Test MSE, shape (1 + n_lambdas, n_degrees); row 0 is OLS.
    r2_vals : np.ndarray            Test R^2, same shape.
    parameters : list               Ridge coefficients for each degree at [ridge] lmbda
                                    (for the parameter plot).
    """
    ridge_cfg = cfg["ridge"]
    lambda_vals = ridge_cfg["lambda_vals"]
    degrees = range(1, ridge_cfg["max_degree"] + 1)
    runge = make_model(cfg)

    mse_vals = np.zeros((1 + len(lambda_vals), len(degrees)))
    r2_vals = np.zeros((1 + len(lambda_vals), len(degrees)))

    # row 0: OLS as the reference (lambda = 0)
    mse_vals[0], r2_vals[0], _ = runge.evaluate_ols_multiple_degrees(degrees)

    # rows 1, 2, ...: Ridge for each lambda
    for i, lmbda in enumerate(lambda_vals):
        mse_vals[i + 1], r2_vals[i + 1], _ = runge.evaluate_ridge_multiple_degrees(
            degrees, lmbda=lmbda
        )

    # coefficients over degree at one lambda, for the parameter plot
    _, _, parameters = runge.evaluate_ridge_multiple_degrees(
        degrees, lmbda=ridge_cfg["lmbda"]
    )

    return degrees, lambda_vals, mse_vals, r2_vals, parameters


def run_ridge(cfg: dict) -> None:
    """
    Run the complete Ridge regression analysis for task b.
    Two experiments are run:

    1. Vary lambda while keeping the polynomial degree fixed.
    2. Vary the polynomial degree while keeping lambda fixed.

    The resulting MSE and R^2 scores are plotted together with the
    fitted regression coefficients for both experiments.

    Parameters
    ----------
    cfg : dict      Configuration dictionary containing the Ridge regression,
                    data-generation, and plotting settings.
    """
    lambda_vals, mse_lambda, r2_lambda, parameters_lambda = run_ridge_over_lambda(cfg)
    degrees, lmbda_vals, mse_degree, r2_degree, parameters_degree = (
        run_ridge_over_degree(cfg)
    )

    plot.plot_parameters(
        lambda_vals, parameters_lambda, variable="lambda", method="ridge", split_at=1e-4
    )
    plot.plot_parameters(degrees, parameters_degree, variable="degree", method="ridge")

    plot.plot_mse_and_r2(
        degrees,
        mse_degree,
        r2_degree,
        variable="degree",
        method="ridge",
        labels=["OLS"] + [rf"Ridge, $\lambda = {lmbda:.0e}$" for lmbda in lmbda_vals],
        suffix="_different_lambda",
    )

    plot.plot_mse_and_r2(
        lambda_vals,
        mse_lambda,
        r2_lambda,
        variable="lambda",
        method="ridge",
        labels=None,  # one curve; the degree is in the filename
        suffix=f"_degree_{cfg['ridge']['degree']}",
    )

    plot.plot_shrinkage_factors(
        singular_values(cfg), cfg["ridge"]["lambda_vals"], cfg["ridge"]["degree"]
    )
    return None
