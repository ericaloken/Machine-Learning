# data.py

import numpy as np
from sklearn.model_selection import train_test_split


def runge_function(x):
    """
    Evaluate the Runge function.

    Parameters
    ----------
    x : float or np.ndarray    Input values.

    Returns
    -------
    float or np.ndarray        Runge function values.
    """
    f = 1 / (1 + 25 * x**2)

    return f


def generate_data(n_points, a=-1, b=1, sigma=0.1, random_state=67, sampling="uniform"):
    """
    Generate noisy data from the Runge function.

    Parameters
    ----------
    n_points : int        Number of data points.
    a : float             Lower bound of the interval.
    b : float             Upper bound of the interval.
    sigma : float         Standard deviation of the noise.
    random_state : int    Random seed.
    sampling : str        Sampling method, "uniform" or "linspace".

    Returns
    -------
    tuple                  Input values and noisy target values.
    """
    np.random.seed(random_state)

    if sampling == "linspace":
        x = np.linspace(a, b, n_points)
    elif sampling == "uniform":
        x = np.random.uniform(a, b, size=n_points)
    else:
        raise ValueError(f"Unknown sampling method. Use linspace or uniform")

    noise = np.random.normal(0, sigma, size=np.shape(x))  # random normal noise
    y = runge_function(x) + noise

    return x, y


def design_matrix(x, degree):
    """
    Create a design matrix for polynomial regression.

    Parameters
    ----------
    x : np.ndarray    Input values.
    degree : int      Polynomial degree.

    Returns
    -------
    np.ndarray        Design matrix.
    """
    X = np.vander(x.flatten(), degree + 1, increasing=True)  # create Vandermode matrix

    return X


def split_data(X, y, test_size=0.2, random_state=None):
    """
    Split data into training and test sets.

    Parameters
    ----------
    X : np.ndarray        Design matrix.
    y : np.ndarray        Target values.
    test_size : float     Fraction of data used for testing.
    random_state : int    Random seed.

    Returns
    -------
    tuple                  X_train, X_test, y_train, and y_test.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
