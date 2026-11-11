# losses.py
"""
Loss functions for neural networks.

Each loss is implemented as a class with a forward method
that returns the scalar loss value over the batch and a
derivative method that returns the gradient of the loss with respect to the predictions.

We also use abstractmethods to enforce that subclasses implement the forward and derivative methods.
"""

import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    """
    Base class for loss functions.

    Subclasses must implement :meth:`forward` and :meth:`derivative`.

    Methods
    -------
    forward(y_true, y_pred)
        Evaluate the scalar loss.
    derivative(y_true, y_pred)
        Evaluate the gradient of the loss with respect to `y_pred`.
    """

    @abstractmethod
    def forward(self, y_true, y_pred):
        """
        Evaluate the loss.

        Parameters
        ----------
        y_true : jax.Array
            Target values.
        y_pred : jax.Array
            Model predictions, same shape as `y_true`.

        Returns
        -------
        jax.Array
            Scalar loss averaged over the batch.
        """

    @abstractmethod
    def derivative(self, y_true, y_pred):
        """
        Evaluate the gradient of the loss with respect to the predictions.

        Parameters
        ----------
        y_true : jax.Array
            Target values.
        y_pred : jax.Array
            Model predictions, same shape as `y_true`.

        Returns
        -------
        jax.Array
            Gradient dL/dy_pred, same shape as `y_pred`.

        """


class MSE(Loss):
    """
    Mean squared error loss for regression tasks.

    L = mean((y_true - y_pred)^2)

    The mean is taken over all elements, i.e. over both samples and outputs.
    """

    def forward(self, y_true, y_pred):
        """
        Evaluate the mean squared error.

        Parameters
        ----------
        y_true : jax.Array
            Target values.
        y_pred : jax.Array
            Model predictions, same shape as `y_true`.

        Returns
        -------
        jax.Array
            Scalar mean squared error.
        """
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true, y_pred):
        """
        Evaluate the gradient of the mean squared error.

        dL/dy_pred = 2 * (y_pred - y_true) / N,

        where N is the total number of elements in `y_true`.

        Parameters
        ----------
        y_true : jax.Array
            Target values.
        y_pred : jax.Array
            Model predictions, same shape as `y_true`.

        Returns
        -------
        jax.Array
            Gradient with respect to `y_pred`, same shape as `y_pred`.
        """
        return 2.0 * (y_pred - y_true) / y_true.size


# Maybe make eps a parameter so change in toml later


class CrossEntropy(Loss):
    """
    Multiclass cross-entropy loss for classification tasks.

    L = -(1/n) * sum_i sum_k y_true[i, k] * log(y_pred[i, k])

    where n is the number of samples. `y_true` is expected to be one-hot
    encoded and `y_pred` to contain class probabilities, e.g. the output of
    a softmax layer. Predictions are clipped to [eps, 1 - eps] with
    eps = 1e-12 to avoid log(0).
    """

    def forward(self, y_true, y_pred):
        """
        Evaluate the multiclass cross-entropy.

        Parameters
        ----------
        y_true : jax.Array
            One-hot encoded targets of shape ``(n_samples, n_classes)``.
        y_pred : jax.Array
            Predicted class probabilities, same shape as `y_true`.

        Returns
        -------
        jax.Array
            Scalar cross-entropy averaged over samples.
        """
        eps = 1e-12
        p = np.clip(y_pred, eps, 1.0 - eps)  # ensure numerical stability
        return -np.mean(np.sum(y_true * np.log(p), axis=-1))

    def derivative(self, y_true, y_pred):
        """
        Evaluate the gradient of the cross-entropy with respect to `y_pred`.

        dL/dy_pred = -y_true / (n * y_pred),

        where n is the number of samples.

        Parameters
        ----------
        y_true : jax.Array
            One-hot encoded targets of shape ``(n_samples, n_classes)``.
        y_pred : jax.Array
            Predicted class probabilities, same shape as `y_true`.

        Returns
        -------
        jax.Array
            Gradient with respect to `y_pred`, same shape as `y_pred`.

        Notes
        -----
        This is the gradient with respect to the probabilities, not the
        softmax pre-activations. When the output layer is softmax, the
        combined gradient with respect to the pre-activations simplifies to
        (y_pred - y_true) / n, which is more stable and should be used
        instead, since ``Softmax.derivative`` is not implemented.
        """
        eps = 1e-12
        p = np.clip(y_pred, eps, 1.0 - eps)
        return -(y_true / p) / y_true.shape[0]


class BinaryCrossEntropy(Loss):
    """
    Binary cross-entropy loss for binary classification tasks.

    L = -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))

    `y_true` is expected to contain labels in {0, 1} and `y_pred` the
    predicted probability of the positive class, e.g. the output of a
    sigmoid layer. Predictions are clipped to [eps, 1 - eps] with
    eps = 1e-12 to avoid log(0).
    """

    def forward(self, y_true, y_pred):
        """
        Evaluate the binary cross-entropy.

        Parameters
        ----------
        y_true : jax.Array
            Binary targets.
        y_pred : jax.Array
            Predicted probabilities, same shape as `y_true`.

        Returns
        -------
        jax.Array
            Scalar binary cross-entropy averaged over all elements.
        """
        eps = 1e-12
        p = np.clip(y_pred, eps, 1.0 - eps)

        return -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))

    def derivative(self, y_true, y_pred):
        """
        Evaluate the gradient of the binary cross-entropy.

        dL/dy_pred = (y_pred - y_true) / (y_pred * (1 - y_pred) * N),

        where N is the total number of elements in `y_true`.

        Parameters
        ----------
        y_true : jax.Array
            Binary targets.
        y_pred : jax.Array
            Predicted probabilities, same shape as `y_true`.

        Returns
        -------
        jax.Array
            Gradient with respect to `y_pred`, same shape as `y_pred`.

        Notes
        -----
        Combined with a sigmoid output layer, the gradient with respect to
        the pre-activations simplifies to (y_pred - y_true) / N.
        """
        eps = 1e-12
        p = np.clip(y_pred, eps, 1.0 - eps)

        return ((p - y_true) / (p * (1.0 - p))) / y_true.size
