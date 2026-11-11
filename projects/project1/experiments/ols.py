# tasks/ols.py

# --------------------------------------------
# Task a: OLS regression
# - Model performance as a function of polynomial degree.
# - Repeat the degree analysis for different numbers of data points.
# - Repeat the degree analysis for different noise levels.
# - Plot MSE, R^2, and fitted coefficients.
# - Additionally plot fitted curves for selected polynomial degrees and data set sizes.
# --------------------------------------------

from .common import make_model
import plotting as plot
import numpy as np
from data import design_matrix, split_data, runge_function
from models import Model


def run_ols_over_n(cfg: dict):
    """
    Evaluate OLS regression over polynomial degree for different data-set sizes.
    The analysis is repeated for different number of data points n_points.

    Parameters
    ----------
    cfg : dict              Configuration dictionary containing settings and
                            parameters required to construct the model.
    Returns
    -------
    degrees : range         Polynomial degrees included in the analysis.
    mse_vals : ndarray      MSE values for each number of data points and polynomial degree.
    r2_vals : ndarray       R^2 scores for each number of data points and polynomial degree.
    """
    ols_cfg = cfg["ols"]
    n_vals = ols_cfg["n_vals"]

    max_degree = ols_cfg["max_degree"]
    degrees = range(1, max_degree + 1)

    mse_vals, r2_vals = np.zeros((len(n_vals), len(degrees))), np.zeros(
        (len(n_vals), len(degrees))
    )
    for i, n in enumerate(n_vals):
        runge_n = make_model(cfg, n_points=n)
        mse_vals[i], r2_vals[i], _ = runge_n.evaluate_ols_multiple_degrees(degrees)

    return degrees, mse_vals, r2_vals


def run_ols_over_sigma(cfg: dict):
    """
    Evaluate OLS regression over polynomial degree for different noise levels.
    The analysis is repeated for differemt noise values.

    Parameters
    ----------
    cfg : dict              Configuration dictionary containing settings and
                            parameters required to construct the model.
    Returns
    -------
    degrees : range         Polynomial degrees included in the analysis.
    mse_vals : ndarray      MSE values for each noise level and polynomial degree.
    r2_vals : ndarray       R^2 scores for each noise level and polynomial degree.
    """
    ols_cfg = cfg["ols"]
    sigma_vals = ols_cfg["sigma_vals"]

    max_degree = ols_cfg["max_degree"]
    degrees = range(1, max_degree + 1)

    mse_vals, r2_vals = np.zeros((len(sigma_vals), len(degrees))), np.zeros(
        (len(sigma_vals), len(degrees))
    )
    for i, _ in enumerate(sigma_vals):
        runge_sigma = make_model(cfg, sigma=sigma_vals[i])
        mse_vals[i], r2_vals[i], _ = runge_sigma.evaluate_ols_multiple_degrees(degrees)
    return degrees, mse_vals, r2_vals


def run_ols_fits(cfg: dict) -> None:
    """
    Fitted OLS curves against the data for several polynomial degrees (columns) and
    data set sizes (rows), together with the true Runge function. Uses the same data,
    train/test split and seed as the rest of the analysis. Made with Claude Sonnet 5.

    Parameters
    ----------
    cfg : dict      Configuration dictionary; [ols] fit_n_vals and fit_degrees select the panels.
    """
    ols_cfg = cfg["ols"]
    grid = np.linspace(-1, 1, 400)  # where the fitted curves are evaluated
    f_grid = runge_function(grid)

    panels = []
    for n in ols_cfg["fit_n_vals"]:
        runge = make_model(cfg, n_points=n)
        x_train, x_test, y_train, y_test = split_data(
            runge.x,
            runge.y,
            test_size=cfg["data"]["test_size"],
            random_state=cfg["data"]["seed"],
        )
        for degree in ols_cfg["fit_degrees"]:
            model = Model.fit(design_matrix(x_train, degree), y_train, method="ols")
            curve = model.predict(design_matrix(grid, degree))
            panels.append(
                {
                    "title": rf"$n={n}$, $d={degree}$",
                    "x_train": x_train,
                    "y_train": y_train,
                    "x_test": x_test,
                    "y_test": y_test,
                    "curves": [("OLS fit", "C0", curve)],
                }
            )

    plot.plot_fit_panels(
        grid,
        f_grid,
        panels,
        "ols_fits_over_degree.pdf",
        ncols=len(ols_cfg["fit_degrees"]),
    )


def run_ols(cfg: dict) -> None:
    """
    Run the complete OLS regression analysis for task a.
    Two experiments are run:

    1. Performance as a function of polynomial degree for different numbers of data points.
    2. Performance as a function of polynomial degree for different noise levels.

    The fitted regression coefficients are also evaluated as a
    function of polynomial degree. The resulting MSE and R^2
    scores and fitted coefficients are plotted.

    Parameters
    ----------
    cfg : dict      Configuration dictionary containing the OLS, data-generation,
                    and plotting settings.
    """
    max_degree = cfg["ols"]["max_degree"]
    degrees = range(1, max_degree + 1)
    parameters = make_model(cfg).evaluate_ols_multiple_degrees(degrees)[
        2
    ]  # gives parameters for each degree
    degrees_n, mse_n, r2_n = run_ols_over_n(cfg)
    degrees_sigma, mse_sigma, r2_sigma = run_ols_over_sigma(cfg)

    n_vals = cfg["ols"]["n_vals"]
    sigma_vals = cfg["ols"]["sigma_vals"]

    # plot
    plot.plot_mse_and_r2(
        degrees_n,
        mse_n,
        r2_n,
        variable="degree",
        method="ols",
        labels=[rf"$n={n}$" for n in n_vals],
        suffix="_different_n",
    )
    plot.plot_mse_and_r2(
        degrees_sigma,
        mse_sigma,
        r2_sigma,
        variable="degree",
        method="ols",
        labels=[rf"$\sigma={s}$" for s in sigma_vals],
        suffix="_different_sigma",
    )

    plot.plot_parameters(
        degrees, parameters, variable="degree", method="ols", split_at=10
    )
    run_ols_fits(cfg)
