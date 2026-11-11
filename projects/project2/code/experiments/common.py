# common.py
"""
Shared utilities for experiments: output paths and result saving.

Attributes
----------
_PROJECT_ROOT : pathlib.Path
    Root directory of the project, two levels above this file.
_OUTPUT_DIR : pathlib.Path
    Top-level directory where all experiment output is written.
"""

import numpy as np
from pathlib import Path
from losses import MSE
from data import generate_data, split_data, Scaler
from activations import Sigmoid, Linear
from network import build_network
from optimizers import make_optimizer

# Project directory
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_DIR = _PROJECT_ROOT / "output"

# ------------------------------------------------------------------
#       Saving and loading results
# ------------------------------------------------------------------


def make_output_dirs(task):
    """
    Create figure and result directories for an experiment.

    The directories are created as ``OUTPUT_DIR/<task>/figures`` and
    ``OUTPUT_DIR/<task>/results``. Existing directories are left unchanged.

    Parameters
    ----------
    task : str
        Name of the experiment, used as the subdirectory name.

    Returns
    -------
    fig_dir : pathlib.Path
        Directory for saving figures.
    result_dir : pathlib.Path
        Directory for saving numerical results.
    """
    task_dir = _OUTPUT_DIR / task

    fig_dir = task_dir / "figures"
    result_dir = task_dir / "results"

    fig_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    return fig_dir, result_dir


def save_results(results, filename):
    """
    Save a dictionary of arrays to an uncompressed ``.npz`` file.

    Each key in `results` becomes the name of an array in the archive.

    Parameters
    ----------
    results : dict of str to array_like
        Arrays to save, keyed by the name they will have in the file.
        Example: ``{"loss": loss_array, "accuracy": accuracy_array}``.
    filename : str or pathlib.Path
        Output path. If it does not end in ``.npz``, NumPy appends the
        extension.

    Returns
    -------
    None
    """
    np.savez(filename, **results)


def load_results(filename):
    """
    Load a ``.npz`` file saved by :func:`save_results` as a dictionary of arrays.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the ``.npz`` file to load.

    Returns
    -------
    dict of str to array_like
        Dictionary containing the arrays from the file.
    """
    with np.load(filename) as f:
        return {key: f[key] for key in f.files}


# ----------------
# Data generation and preprocessing
# ----------------


def get_runge_data(cfg, n_points=None):
    """
    Generate or load data based on the configuration.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary specifying the data source and parameters.

    Returns
    -------
    X_train, X_test, y_train, y_test : array_like
        Training and testing data splits.
    """
    data_cfg = cfg["data"]
    if n_points is None:
        n_points = data_cfg["n_points"]

    x, y = generate_data(
        n_points,
        sigma=data_cfg["sigma"],
        random_state=data_cfg["seed"],
        sampling=data_cfg["x_sampling"],
    )
    return split_data(
        x, y, test_size=data_cfg["test_size"], random_state=data_cfg["seed"]
    )


# ----------------
# Neural network training
# ----------------


def train_network(
    cfg,
    layers,
    optimizer_name,
    gamma,
    x_train,
    y_train,
    hidden_activation=Sigmoid,
    regularizer=None,
):
    """
    Train one network with the given configuration.

    Parameters
    ----------
    cfg : dict              Configuration dictionary.
    layers : list[int]      Nodes in each layer, e.g. [1, 50, 50, 1].
    optimizer_name : str    "gd", "rmsprop" or "adam".
    gamma : float           Learning rate.
    x_train, y_train        Training data (unscaled).
    hidden_activation : class      Activation class for the hidden layers, e.g. Sigmoid or ReLU.
    regularizer : Regularizer      Optional L1 or L2 penalty, e.g. L2(lmbda).

    Returns
    -------
    net : NeuralNetwork     Trained network.
    scaler : Scaler         Scaler fitted on the training data.
    history : np.ndarray    Training loss (scaled y) after each epoch.

    """
    scaler = Scaler(x_train, y_train)  # fit scaler on training data

    activations = [hidden_activation() for _ in layers[1:-1]] + [Linear()]
    net = build_network(
        layers,
        activations,
        MSE(),
        make_optimizer(optimizer_name, gamma),
        regularizer=regularizer,
        random_state=cfg["data"]["seed"],
    )

    if optimizer_name == "gd":
        batch_size = None
    else:
        batch_size = cfg["training"]["batch_size"]

    history = net.fit(
        scaler.x(x_train),
        scaler.y(y_train),
        epochs=cfg["training"]["epochs"],
        batch_size=batch_size,
    )
    history = np.asarray(history, dtype=float)  # convert to array for np saving
    return net, scaler, history


def predict(net, scaler, x):
    """Network prediction on the original scale of y."""
    return scaler.inverse_y(net.predict(scaler.x(x)))
