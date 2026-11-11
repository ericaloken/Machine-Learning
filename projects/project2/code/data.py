# data.py

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# for importing MNIST from the web
import urllib.request

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


"""
From ../project1/data.py, where function docstrings can be found
"""


def runge_function(x):
    """
    Runge function: f(x) = 1 / (1 + 25 * x^2)

    x can be a scalar or an array.
    Parameters:
    ----------
    x : array_like
        Input values.

    Returns:
    -------
    f : array_like
        Output values of the Runge function evaluated at x.
    """
    f = 1 / (1 + 25 * x**2)

    return f


def generate_data(n_points, a=-1, b=1, sigma=0.1, random_state=67, sampling="uniform"):
    """
    docstring needed
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
    """Create Vandermonde matrix."""
    return np.vander(x.flatten(), degree + 1, increasing=True)


def split_data(X, y, test_size=0.2, random_state=None):
    """Split data into train and test sets with chosen test_size."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def load_mnist(path="mnist.npz"):
    """
    Load MNIST, flattened to 784 features and scaled to [0, 1].

    A relative `path` is resolved from the code/ directory (where this file is),
    so the script can be run from any working directory.
    """
    path = Path(path)
    if not path.is_absolute():
        path = Path(__file__).parent / path

    if not path.exists():
        print(f"Downloading MNIST to {path}")
        urllib.request.urlretrieve(MNIST_URL, path)

    with np.load(path) as f:
        X_train, y_train = f["x_train"], f["y_train"]
        X_test, y_test = f["x_test"], f["y_test"]

    X_train = X_train.reshape(-1, 784) / 255.0
    X_test = X_test.reshape(-1, 784) / 255.0

    return X_train, y_train, X_test, y_test


class Scaler:
    """
    Standardization with training statistics using sklearn's StandardScaler.
    This class is used to scale the data before training and testing the model.
    Used for neural network, as for linear regression, Model.fit() can handle unscaled data.

    Reshaping to (-1, 1) is necessary for sklearn's StandardScaler to work correctly with 1D arrays.
    """

    def __init__(self, x_train, y_train):
        self._x = StandardScaler().fit(np.reshape(x_train, (-1, 1)))
        self._y = StandardScaler().fit(np.reshape(y_train, (-1, 1)))

    def x(self, x):
        return self._x.transform(np.reshape(x, (-1, 1)))

    def y(self, y):
        return self._y.transform(np.reshape(y, (-1, 1)))

    def inverse_y(self, y_scaled):
        y_scaled = np.reshape(
            np.asarray(y_scaled), (-1, 1)
        )  # in case y_scaled is not already a 2D array
        return self._y.inverse_transform(y_scaled).ravel()
