# metrics.py

import numpy as np


def mse(y, y_tilde) -> float:
    """
    Compute the mean squared error.

    Parameters
    ----------
    y : np.ndarray        True values.
    y_tilde : np.ndarray  Predicted values.

    Returns
    -------
    float                 Mean squared error.
    """
    N = len(y)
    MSE = np.sum((y - y_tilde) ** 2) / N

    return MSE


def rel_error(x, y) -> float:
    """
    Compute relative parameter error between two parameter vectors.
    """
    return np.linalg.norm(x - y) / np.linalg.norm(y)


def r2(y, y_tilde) -> float:
    """
    Compute the coefficient of determination, R2.

    Parameters
    ----------
    y : np.ndarray        True values.
    y_tilde : np.ndarray  Predicted values.

    Returns
    -------
    float                 Coefficient of determination.
    """
    R2 = 1 - np.sum((y - y_tilde) ** 2) / np.sum((y - np.mean(y)) ** 2)

    return R2
