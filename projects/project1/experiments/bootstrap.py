# tasks/bootstrap.py

# ----------------------------------------------------------------------
# Task c: Bootstrap bias-variance analysis
# - Training vs test MSE over polynomial degree
# - Estimate bias, variance, and MSE using bootstrap resampling
# - Compare bias estimates using y and the true function f as reference
# - Bias-variance tradeoff for different numbers of data points
# -----------------------------------------------------------------------

import numpy as np
from .common import make_model
import plotting as plot


def run_train_test_mse(cfg: dict):
    """
    Evaluate training and test MSE over polynomial degree.

    Parameters
    ----------
    cfg : dict                  Configuration dictionary containing the bootstrap and
                                data-generation settings.
    Returns
    -------
    degrees : range             Polynomial degrees included in the analysis.
    train_mse : array-like      Training MSE for each polynomial degree.
    test_mse : array-like       Test MSE for each polynomial degree.
    """
    bootstrap_cfg = cfg["bootstrap"]
    n_points = bootstrap_cfg["n_points"]
    max_degree = bootstrap_cfg["max_degree"]
    degrees = range(1, max_degree + 1)

    runge = make_model(cfg, n_points=n_points)
    train_mse, test_mse = runge.train_test_mse_vs_degree(degrees=degrees)
    return degrees, train_mse, test_mse


def run_bias_variance(cfg: dict, reference="y", linreg_method="ols", lmbda=None):
    """
    Perform bootstrap bias-variance analysis over polynomial degree.
    The bias can be evaluated relative to either the noisy observations
    y or the true Runge function f.

    Parameters
    ----------
    cfg : dict                          Configuration dictionary containing the bootstrap and
                                        data-generation settings.
    reference : {"y", "f"}, optional    Reference used when computing the bias. Default is "y".
    linreg_method : str, optional       Regression method used in the bootstrap analysis. Default is "ols".
    lmbda : float, optional             Regularization strength for Ridge regression. Default is None.

    Returns
    -------
    degrees : range                     Polynomial degrees included in the analysis.
    bias2 : array-like                  Estimated squared bias for each polynomial degree.
    variance : array-like               Estimated model variance for each polynomial degree.
    mse : array-like                    Estimated MSE for each polynomial degree.
    """
    bootstrap_cfg = cfg["bootstrap"]
    n_points = bootstrap_cfg["n_points"]
    max_degree = bootstrap_cfg["max_degree"]
    degrees = range(1, max_degree + 1)
    n_bootstrap = bootstrap_cfg["n_bootstrap"]
    runge = make_model(cfg, n_points=n_points)

    bias2, variance, mse = runge.bias_variance_bootstrap(
        degrees=degrees,
        n_bootstrap=n_bootstrap,
        bias_reference=reference,
        method=linreg_method,
        lmbda=lmbda,
    )
    return degrees, bias2, variance, mse


def run_bias_variance_over_n(cfg: dict, reference="y", linreg_method="ols", lmbda=None):
    """
    Perform bootstrap bias-variance analysis for different data-set sizes.

    Parameters
    ----------
    cfg : dict                          Configuration dictionary containing the bootstrap and
                                        data-generation settings.
    reference : {"y", "f"}, optional    Reference used when computing the bias. Default is "y".
    linreg_method : str, optional       Regression method used in the bootstrap analysis. Default is "ols".
    lmbda : float, optional             Regularization strength for Ridge regression. Default is None.

    Returns
    -------
    n_vals : array-like                 Numbers of data points included in the analysis.
    degrees : range                     Polynomial degrees included in the analysis.
    bias2_vals : ndarray                Squared bias for each data-set size and polynomial degree.
    variance_vals : ndarray             Variance for each data-set size and polynomial degree.
    mse_vals : ndarray                  MSE for each data-set size and polynomial degree.
    """
    bootstrap_cfg = cfg["bootstrap"]
    n_vals = bootstrap_cfg["n_vals"]
    max_degree = bootstrap_cfg["max_degree"]
    degrees = range(1, max_degree + 1)
    n_bootstrap = bootstrap_cfg["n_bootstrap"]

    bias2_vals = np.zeros((len(n_vals), len(degrees)))
    variance_vals = np.zeros((len(n_vals), len(degrees)))
    mse_vals = np.zeros((len(n_vals), len(degrees)))

    for i, n_points in enumerate(n_vals):
        runge = make_model(cfg, n_points=n_points)
        bias2_vals[i], variance_vals[i], mse_vals[i] = runge.bias_variance_bootstrap(
            degrees=degrees,
            n_bootstrap=n_bootstrap,
            bias_reference=reference,
            method=linreg_method,
            lmbda=lmbda,
        )
    return n_vals, degrees, bias2_vals, variance_vals, mse_vals


def run_bootstrap(cfg: dict) -> None:
    """
    Run the complete bootstrap analysis for task c.
    Three analyses are performed:

    1. Training and test MSE as functions of polynomial degree.
    2. Bootstrap bias-variance analysis using both y and the true
       function f as the bias reference.
    3. Bootstrap bias-variance analysis for different numbers of
       data points.

    The resulting training and test errors, squared bias, variance,
    and MSE values are plotted.

    Parameters
    ----------
    cfg : dict          Configuration dictionary containing the bootstrap,
                        data-generation, regression, and plotting settings.
    """
    # training vs test MSE
    linreg_method = cfg["bootstrap"]["linreg_method"]
    sigma = cfg["data"]["sigma"]
    degrees, train_mse, test_mse = run_train_test_mse(cfg)
    plot.plot_train_test_mse(degrees, train_mse, test_mse, sigma=sigma)

    # bias-variance with y and f as reference
    seed = cfg["data"]["seed"]
    np.random.seed(seed)
    degrees, bias_y, variance, mse = run_bias_variance(cfg, "y", linreg_method)
    np.random.seed(seed)
    _, bias_f, variance_f, mse_f = run_bias_variance(cfg, "f", linreg_method)

    # check that MSE and variance does not depend on the bias reference, and that MSE = bias + variance
    assert np.allclose(variance, variance_f) and np.allclose(mse, mse_f)
    assert np.allclose(mse, bias_y + variance)

    plot.plot_bias_variance_both(
        degrees, mse, variance, bias_f, bias_y, sigma=sigma, method=linreg_method
    )

    # bias-variance for several n_points
    n_vals, degrees, bias_n, variance_n, mse_n = run_bias_variance_over_n(
        cfg, linreg_method=linreg_method
    )
    plot.plot_bias_variance_multiple(
        degrees,
        bias_n,
        variance_n,
        mse_n,
        labels=[rf"$n={n}$" for n in n_vals],
        varying_parameter="n",
        method=linreg_method,
        suffix="_bootstrap",
    )

    # noise floor
    plot.plot_noise_floor(
        degrees, mse_f, bias_f, variance_f, bias_y, sigma=sigma, method=linreg_method
    )
