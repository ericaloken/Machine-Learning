# activations.py
"""
Activation functions for neural networks.
"""

import numpy as np
from abc import ABC, abstractmethod


class Activation(ABC):
    """
    Base class for activation functions.

    Subclasses must implement the :meth:`forward` and :meth:`derivative` methods.

    Methods
    -------
    forward(z)
        Compute the activation function for input z.
    derivative(z)
        Compute the derivative of the activation function for input z.
    """

    @abstractmethod
    def forward(self, z):
        """
        Evaluate the activation function.

        Parameters
        ----------
        z : array-like
            Input to the activation function.

        Returns
        -------
        jax.Array
            Output of the activation function.

        """

    @abstractmethod
    def derivative(self, z):
        """
        Evaluate the derivative of the activation function.

        Parameters
        ----------
        z : jax.Array
            Input to the activation function.

        Returns
        -------
        jax.Array
            Derivative of the activation function.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError


class Sigmoid(Activation):
    """
    Logistic sigmoid activation function.

    sigma(z) = 1 / (1 + exp(-z))

    The output lies in (0,1).
    """

    def forward(self, z):
        """
        Evaluate the sigmoid function element-wise.

        Parameters
        ----------
        z : jax.Array
            Input to the sigmoid function.

        Returns
        -------
        jax.Array
            Output of the sigmoid function.
        """
        with np.errstate(over="ignore"):
            return 1 / (1 + np.exp(-z))

    def derivative(self, z):
        """
        Evaluate the derivative of the sigmoid function element-wise.

        Uses identity: sigma'(z) = sigma(z) * (1 - sigma(z))

        Parameters
        ----------
        z : jax.Array
            Input to the sigmoid function.

        Returns
        -------
        jax.Array
            Derivative of the sigmoid function.
        """
        a = self.forward(z)
        return a * (1 - a)


class ReLU(Activation):
    """
    Rectified Linear Unit (ReLU) activation function.

    ReLU(z) = max(0, z)
    """

    def forward(self, z):
        """
        Evaluate the ReLU function element-wise.

        Parameters
        ----------
        z : jax.Array
            Input to the ReLU function.

        Returns
        -------
        jax.Array
            Output of the ReLU function.
        """
        return np.maximum(z, 0)

    def derivative(self, z):
        """
        Evaluate the derivative of the ReLU function element-wise.

        Parameters
        ----------
        z : jax.Array
            Input to the ReLU function.

        Returns
        -------
        jax.Array
            Derivative of the ReLU function.
        """
        return (z > 0).astype(z.dtype)


class LeakyReLU(Activation):
    """
    Leaky Rectified Linear Unit (Leaky ReLU) activation function.

    LeakyReLU(z) = z if z > 0 else alpha * z

    Parameters
    ----------
    alpha : float, optional
        Slope for negative inputs. Default is 0.01.
    """

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, z):
        """
        Evaluate the Leaky ReLU function element-wise.

        Parameters
        ----------
        z : jax.Array
            Input to the Leaky ReLU function.

        Returns
        -------
        jax.Array
            Output of the Leaky ReLU function.
        """
        return np.maximum(z, self.alpha * z)

    def derivative(self, z):

        return (z > 0).astype(z.dtype) + self.alpha * (z <= 0).astype(z.dtype)


class Linear(Activation):
    """
    Linear activation function (identity function).

    f(z) = z
    """

    def forward(self, z):
        """
        Return unchanged input.

        Parameters
        ----------
        z : jax.Array
            Input to the linear function.

        Returns
        -------
        jax.Array
            Output of the linear function (same as input).
        """
        return z

    def derivative(self, z):
        """
        Derivative of the linear function is constant 1.

        Parameters
        ----------
        z : jax.Array
            Input to the linear function.

        Returns
        -------
        jax.Array
            Derivative of the linear function (constant 1).
        """
        return 1.0


class Softmax(Activation):
    """
    Softmax activation over the last axis.

    softmax(z)_i = exp(z_i) / sum_j exp(z_j)

    Typically used in the output layer for multiclass classification.
    """

    def forward(self, z):
        """
        Evaluate the softmax function along the last axis.

        The maximum along the last axis is subtracted before exponentiating
        for numerical stability; this does not change the result.

        Parameters
        ----------
        z : jax.Array
            Pre-activation values of shape ``(..., n_classes)``.

        Returns
        -------
        jax.Array
            Class probabilities of the same shape as `z`, summing to 1
            along the last axis.
        """
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def derivative(self, z):
        """
        Not implemented for softmax.

        The softmax derivative is a full Jacobian rather than an element-wise
        function. When softmax is combined with cross-entropy loss, the
        gradient with respect to `z` simplifies to ``softmax(z) - y``, which
        should be handled by the loss function instead.

        Parameters
        ----------
        z : jax.Array
            Pre-activation values.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "Softmax derivative is not implemented. Use cross-entropy loss instead."
        )
