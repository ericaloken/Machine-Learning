# metric.py

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

def accuracy(y_true, y_pred) -> float:
    """Fraction of correct class labels (y_pred as labels, not probabilities)."""
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred, n_classes=10) -> np.ndarray:
    """C[i, j] = number of samples with true class i predicted as class j."""
    C = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(C, (y_true, y_pred), 1)
    return C

def precision_recall_f1(C):
    """
    Precision, recall and F1 score per class from a confusion matrix C
    (rows = true class, columns = predicted class).

    precision_k = C[k, k] / column sum k   (of everything predicted as k, how much is k)
    recall_k    = C[k, k] / row sum k      (of everything that is k, how much was found)
    F1_k        = 2 * precision_k * recall_k / (precision_k + recall_k)

    Returns three arrays of shape (n_classes,). Take .mean() for the macro average.
    """
    tp = np.diag(C)
    precision = tp / C.sum(axis=0)
    recall = tp / C.sum(axis=1)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1
