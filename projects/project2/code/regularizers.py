# regularizers.py
"""
L1 and L2 regularization for linear models and neural networks.
Only the weights are regularized, not the biases, for NN. So ensure that
for NN a list of (W, b) tuples is passed to the regularizer, not just W.

We also use abstractmethods to enforce that subclasses implement the penalty and grad methods.

Lambda convention:
-----------
The penalty is scaled by 1/n, where n is the number of samples
in the full training set (not the mini-batch size).
"""

import numpy as np
from abc import ABC, abstractmethod


def _is_layer_list(params):
    """True if params is a list of (W, b) tuples, i.e. neural network parameters."""
    return isinstance(params, (list, tuple)) and all(
        isinstance(layer, (list, tuple)) and len(layer) == 2 for layer in params
    )


class Regularizer(ABC):
    """
    Base class for all regularizers.

    Subclasses must implement the :meth:`penalty` and :meth:`grad` methods.

    Parameters
    ----------
    lmbda : float
        Regularization strength (lambda).
    """

    def __init__(self, lmbda=0.0):
        self.lmbda = lmbda

    def _weights(self, params):
        """
        List of weights only, biases left out
        """
        if _is_layer_list(params):
            return [W for W, b in params]
        else:
            return [params]  # single weight vector for linear model

    def _grad_weights(self, f, params):
        """Apply function f to weights, zeros for biases. Same structure as params."""
        if _is_layer_list(params):
            return [(f(W), np.zeros_like(b)) for W, b in params]
        return f(params)

    @abstractmethod
    def penalty(self, params, n):
        """
        Evaluate the penalty term for the given parameters.

        Parameters
        ----------
        params : array_like or list of (W, b)
            Model parameters, e.g. ``theta`` for linear models or a list of ``(W, b)`` for neural networks.
            Only the weights are penalized, not the biases.
        n : int
            Number of samples in the full training set.

        Returns
        -------
        float
            Scalar penalty term.
        """

    @abstractmethod
    def grad(self, params, n):
        """
        Evaluate the gradient of the penalty with respect to the parameters.

        Parameters
        ----------
        params : array_like or list of (W, b)
            Model parameters, e.g. ``theta`` or a list of ``(W, b)`` tuples.
        n : int
            Number of samples in the full training set.

        Returns
        -------
        np.ndarray or list of (np.ndarray, np.ndarray)
            Gradient, same structure as `params`, so it can be added to the
            loss gradient. The gradient is zero for the biases.

        """


class L1(Regularizer):
    """
    L1 (Lasso) regularization.

    Penalty: (lmbda/n) * sum(abs(W)) for all weights W.
    Gradient: (lmbda/n) * sign(W) for all weights W (subgradient), zero for biases.
    """

    def penalty(self, params, n):
        total = 0.0

        for W in self._weights(params):
            total += np.sum(np.abs(W))
        return self.lmbda / n * total

    def grad(self, params, n):
        def grad_param(p):
            return self.lmbda / n * np.sign(p)

        return self._grad_weights(grad_param, params)


class L2(Regularizer):
    """
    L2 (Ridge) regularization.

    Penalty: (lmbda/n) * sum W^2 for all weights W.
    Gradient: (2 * lmbda/n) * W for all weights W, zero for biases.
    """

    def penalty(self, params, n):
        total = 0.0
        for W in self._weights(params):
            total += np.sum(W**2)
        return self.lmbda / n * total

    def grad(self, params, n):
        def grad_param(p):
            return 2 * self.lmbda / n * p

        return self._grad_weights(grad_param, params)
