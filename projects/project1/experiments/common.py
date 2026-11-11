# tasks/common.py

# helper functions for tasks

import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

from models import Model
from analysis import RungeRegression
from data import design_matrix
from optimizers import (
    gradient_descent,
    momentum,
    adagrad,
    rmsprop,
    adam,
    stochastic_gradient_descent,
)

# helper functions shared across tasks


def make_model(
    cfg: dict, n_points: int | None = None, sigma: float | None = None
) -> RungeRegression:
    """
    Create a RungeRegression model from the configuration.
    n_points and sigma override the config values when given. (Claude AI's suggestion)

    Parameters
    ----------
    cfg : dict                          Configuration dictionary containing settings.
    n_points : int or None, optional    Number of data points. If None, the value from the config is used.
    sigma : float or None, optional     Standard deviation of the added noise. If None, the value from
                                        the config is used.
    Returns
    -------
    RungeRegression                     Configured RungeRegression model.
    """
    data_cfg = cfg["data"]
    return RungeRegression(
        n_points=data_cfg["n_points"] if n_points is None else n_points,
        sigma=data_cfg["sigma"] if sigma is None else sigma,
        random_state=data_cfg["seed"],
        x_sampling=data_cfg["x_sampling"],
    )


def get_data(runge: RungeRegression):
    """
    Return the input and target data from a RungeRegression model.

    Parameters
    ----------
    runge : RungeRegression     Regression model containing the generated data.

    Returns
    -------
    x : array-like              Input values.
    y : array-like              Corresponding noisy target values.
    """
    return runge.x, runge.y


def get_gd_data(cfg: dict, method: str):
    """
    Build design matrix and target vector for gradient descent analysis.
    """
    if method not in ("ols", "ridge"):
        raise ValueError("Method must be 'ols' or 'ridge'.")

    gd_cfg = cfg["gd"]
    degree = gd_cfg["degree"]

    runge = make_model(cfg)
    X = design_matrix(runge.x, gd_cfg["degree"])
    y = runge.y
    lmbda = gd_cfg["lmbda"] if method == "ridge" else None
    return X, y, lmbda


def logspace(vals: dict) -> np.ndarray:
    """
    Generate logarithmically spaced values from configuration settings.

    Parameters
    ----------
    vals : dict         Dictionary containing ``start``, ``stop``, and ``num``, which
                        define the exponents and number of values passed to np.logspace.
    Returns
    -------
    ndarray             Logarithmically spaced values.
    """
    start, stop, num = vals["start"], vals["stop"], vals["num"]
    return np.logspace(start, stop, num)


OPTIMIZER_NAMES = ["GD", "momentum", "adagrad", "rmsprop", "adam"]


def optimizer_settings(name, gamma, cfg, batch_size=None, schedule=None):
    """
    Set the optimizer function and its parameters based on the optimizer name.

    Parameters
    ----------
    name : str                  Name of the optimizer ("GD", "momentum", "adagrad", "rmsprop", or "adam").
    gamma : float       Learning rate for the optimizer.
    cfg : dict                  Configuration dictionary containing the adaptive gradient descent settings.

    Returns
    -------
    optimizer_func : function   The optimizer function corresponding to the given name.
    optimizer_params : dict     Dictionary of parameters for the optimizer function.
    """

    adaptive_gd_cfg = cfg["adaptive_gd"]
    eps = adaptive_gd_cfg["epsilon"]

    if name == "GD":
        if batch_size is None:
            return gradient_descent, {"gamma": gamma}
        return stochastic_gradient_descent, {  # SGD if batch_size is provided
            "gamma": gamma,
            "batch_size": batch_size,
            "schedule": schedule,
        }

    if name == "momentum":
        beta = adaptive_gd_cfg["momentum_beta"]
        return momentum, {
            "gamma": gamma,
            "beta": beta,
            "batch_size": batch_size,
            "schedule": schedule,
        }

    if name == "adagrad":
        return adagrad, {
            "gamma": gamma,
            "epsilon0": eps,
            "batch_size": batch_size,
            "schedule": schedule,
        }

    if name == "rmsprop":
        rho = adaptive_gd_cfg["rmsprop_beta"]
        return rmsprop, {
            "gamma": gamma,
            "rho": rho,
            "epsilon0": eps,
            "batch_size": batch_size,
            "schedule": schedule,
        }

    if name == "adam":
        return adam, {
            "alpha": gamma,  # adam calls its learning rate alpha
            "beta1": adaptive_gd_cfg["adam_beta1"],
            "beta2": adaptive_gd_cfg["adam_beta2"],
            "epsilon0": eps,
            "batch_size": batch_size,
            "schedule": schedule,
        }

    raise ValueError(
        f"Unknown optimizer name: {name}. Must be one of {OPTIMIZER_NAMES}."
    )


def iterations_for_target(errors, target_error):
    """
    Determine the number of iterations needed to converge to a specified target error,
    defined as the first iteration after which the error stays below the target for
    the rest of the run. Oscillating methods (e.g. Adam and RMSProp with a constant
    learning rate) can dip below the target and then spike above it again, so the
    first crossing alone would overstate how fast they converge.

    Parameters
    ----------
    errors : np.ndarray         Array of relative errors at each iteration.
    target_error : float        Target relative error threshold.

    Returns
    -------
    float : Idx of the last iteration above the target error, plus one. Returns np.nan if the
            last iteration is above the target error, and 1 if the first iteration is below
            the target error.
    """

    errors = np.asarray(errors)  # can accept lists as well as arrays
    above = (errors > target_error) | ~np.isfinite(
        errors
    )  # NaN/inf will count as above

    indices_above = np.where(above)[0]

    if len(errors) == 0 or above[-1]:
        return np.nan  # ends above the target

    if len(indices_above) == 0:
        return 1  # below the target from the first iteration

    return indices_above[-1] + 1  # idx of the last one above


def sklearn_lasso_fit(X_train, y_train, lmbda):
    """
    Fit a Lasso regression model using Scikit-Learn's Lasso implementation.
    """
    mean_X, std_X, mean_y, std_y = Model.training_statistics(X_train, y_train)
    model = Model(np.zeros(X_train.shape[1]), mean_X, std_X, mean_y, std_y)
    X_scaled, y_scaled = model.scale(X_train, y_train)

    alpha = lmbda / (2 * len(y_train))  # Scikit-Learn's alpha parameter convention

    # data is centered so we can set fit_intercept=False
    lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=100000, tol=1e-10)
    # coordinate descent does not converge for high degree and small lambda
    # (ill-conditioned design matrix); this is discussed in the report, so the
    # warning is silenced to keep the output readable
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        lasso.fit(X_scaled[:, 1:], y_scaled)  # skip the bias term in X_scaled

    theta = np.concatenate(([0.0], lasso.coef_))  # 0 for the bias term
    return Model(theta=theta, mean_X=mean_X, std_X=std_X, mean_y=mean_y, std_y=std_y)
