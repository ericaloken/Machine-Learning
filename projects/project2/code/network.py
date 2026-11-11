# network.py
"""
Feed-forward NN with backpropagation.
"""

import numpy as np

from activations import LeakyReLU, ReLU, Softmax
from losses import CrossEntropy
from jax.tree_util import tree_map


class NeuralNetwork:
    """
    Docstring here
    """

    def __init__(
        self, layers, activations, loss, optimizer, regularizer=None, random_state=None
    ):
        """
        Parameters
        ----------
        layers : list[int]
            Number of nodes in each layer, including input and output.

            Example: [1, 50, 50, 1]

        activations : list[Activation]
            Activation object for each layer except the input layer.

        loss : Loss
            Loss function, e.g. MSE().

        optimizer : Optimizer
            Optimizer, e.g. Adam().

        regularizer : Regularizer or None
            Optional L1/L2 regularization.
        """

        if len(layers) - 1 != len(activations):
            raise ValueError(
                "Number of activations must be equal to number of layers - 1."
            )

        self.layers = layers
        self.activations = activations
        self.loss = loss
        self.optimizer = optimizer
        self.regularizer = regularizer
        self._rng = np.random.default_rng(random_state)

        # Network parameters
        self.params = self._initialize_parameters()

        # Cached values for backpropagation
        self._a_cache = []
        self._z_cache = []

    def _initialize_parameters(self):
        """
        Initialize weights using Xavier initialization for Sigmoid and
        linear activations, and He initialization for ReLu and LeakyReLU.
        Biases are initialized to zero.

        Returns
        -------
        list[tuple]
            ``[(W1, b1), (W2, b2), ...]``
        """
        params = []

        for n_in, n_out, activation in zip(
            self.layers[:-1], self.layers[1:], self.activations
        ):
            # He or Xavier initialization
            std = (
                np.sqrt(2.0 / n_in)
                if isinstance(activation, (ReLU, LeakyReLU))
                else np.sqrt(2.0 / (n_in + n_out))
            )

            W = self._rng.normal(loc=0.0, scale=std, size=(n_out, n_in))
            b = np.zeros(n_out)

            params.append((W, b))

        return params

    def _feed_forward(self, params, X):
        """
        Feed forward with given parameters.

        Parameters
        ----------
        params : list[tuple]
            Network parameters ``[(W1, b1), (W2, b2), ...]``.
        X : jax.Array
            Input data, shape (n_samples, n_features).

        Returns:
        -------
        a_list : list[jax.Array]
            List of activations for each layer, including input layer.
        z_list : list[jax.Array]
            List of pre-activation values for each layer, excluding input layer.
        """
        a_list = [X]
        z_list = []

        for (W, b), activation in zip(params, self.activations):
            z = a_list[-1] @ W.T + b
            a = activation.forward(z)

            z_list.append(z)
            a_list.append(a)

        return a_list, z_list

    def forward(self, X):
        """
        Feed-forward pass with the current parameters.

        Adapted version of feed_forward_batch from the week 41 exercises.
        Stores the activations and pre-activations for :meth:`backward`.

        Parameters
        ----------
        X : array_like
            Input data, shape (n_samples, n_features).

        Returns
        -------
        jax.Array
            Network output, shape (n_samples, n_outputs).

        Raises
        ------
        ValueError
            If `X` is not 2D or has the wrong number of features.
        """
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("Input X must be shape (n_samples, n_features).")

        if X.shape[1] != self.layers[0]:
            raise ValueError("Incorrect number of imput features")

        self._a_cache, self._z_cache = self._feed_forward(self.params, X)

        return self._a_cache[-1]  # last activation is the output of the network

    def _output_delta(self, y, y_pred, z):
        """
        Gradient of the loss with respect to the output pre-activation z. Made with help from Claude.

        Softmax with cross-entropy combines to (y_pred - y) / n.
        Otherwise dL/dz = dL/da * f'(z).
        """
        output_activation = self.activations[-1]

        if isinstance(output_activation, Softmax):
            if not isinstance(self.loss, CrossEntropy):
                raise ValueError("A softmax output layer requires CrossEntropy loss.")
            return (y_pred - y) / y.shape[0]

        dL_da = self.loss.derivative(y, y_pred)
        return dL_da * output_activation.derivative(z)

    def backward(self, X, y):
        """
        Compute gradients using back propagation. From week 42 exercises

        Parameters
        ----------
        X : jax.Array
            Input data, shape (n_samples, n_features).
        y : jax.Array
            Target values, shape (n_samples, n_outputs).

        Returns
        -------
        grads : list[tuple]
            List of gradients for each layer, where each element is a tuple (dW, db
        """
        # Forward pass
        y_pred = self.forward(X)

        if y.shape != y_pred.shape:
            raise ValueError("y and y_pred must have the same shape.")

        grads = [None] * len(self.params)

        # Backpropagate trough every layer
        for i in reversed(range(len(self.params))):
            z = self._z_cache[i]
            a_prev = self._a_cache[i]

            # Output or hidden layer
            if i == len(self.params) - 1:
                dL_dz = self._output_delta(y, y_pred, z)
            else:
                W_next, _ = self.params[i + 1]
                dL_dz = (dL_dz @ W_next) * self.activations[i].derivative(z)

            # Parameter gradients
            dL_dW = dL_dz.T @ a_prev
            dL_db = np.sum(dL_dz, axis=0)
            grads[i] = (dL_dW, dL_db)

        return grads

    def _output(self, params, X):
        """
        Network output only. Works like forward(), but does not
        cache anything and each layers activation is computed from the previous layer's output.
        """
        a = np.asarray(X)
        for (W, b), activation in zip(params, self.activations):
            a = activation.forward(a @ W.T + b)
        return a

    def loss_at(self, params, X, y):
        """
        Loss L for the given parameters, without the regularization term.
        """
        return self.loss.forward(y, self._output(params, X))

    def fit(self, X, y, epochs, batch_size):
        """
        Train the network using mini-batch gradient descent.
        If batch_size = None, use full batch gradient descent.

        Returns
        -------
        history : list
            Loss L on the full training set after each epoch, without the
            regularization term.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n = X.shape[0]

        if batch_size is None:
            batch_size = n

        if batch_size <= 0 or batch_size > n:
            raise ValueError(
                "batch_size must be a positive integer less than or equal to the number of samples."
            )

        history = []
        self.optimizer.reset()  # reset optimizer state before training
        for epoch in range(epochs):

            # Shufflle traing samples
            indeces = self._rng.permutation(n)

            for start in range(0, n, batch_size):
                batch_indices = indeces[start : start + batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                grads = self.backward(X_batch, y_batch)
                if (
                    self.regularizer is not None
                ):  # if regularization is used, add the regularization gradients to the parameter gradients
                    # n is the full training size, see regularizers.py
                    reg_grads = self.regularizer.grad(
                        self.params, n
                    )  # regularization gradients

                    def grad_with_reg(grad, reg_grad):
                        # grad and reg_grad are one parameter array each (a W or a b)
                        return grad + reg_grad

                    grads = tree_map(grad_with_reg, grads, reg_grads)

                self.params = self.optimizer.step(self.params, grads)

            # Loss after epoch
            loss = self.loss_at(self.params, X, y)
            history.append(loss)
            if not np.isfinite(loss) or loss > 1e6:  # diverged, no point continuing
                break

        return history

    def predict(self, X):
        return self._output(self.params, X)


def build_network(
    layers, activations, loss, optimizer, regularizer=None, random_state=None
):
    """
    Build a neural network with the given configuration.

    Parameters
    ----------
    layers : list[int]
        Number of nodes in each layer, including input and output.
    activations : list[Activation]
        Activation object for each layer except the input layer.
    loss : Loss
        Loss function, e.g. MSE().
    optimizer : Optimizer
        Optimizer, e.g. Adam().

    Returns
    -------
    NeuralNetwork
        Configured neural network instance.
    """
    return NeuralNetwork(
        layers=layers,
        activations=activations,
        loss=loss,
        optimizer=optimizer,
        regularizer=regularizer,
        random_state=random_state,
    )
