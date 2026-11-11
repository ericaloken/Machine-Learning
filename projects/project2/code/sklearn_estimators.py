# sklearn_estimators.py

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import numpy as np

from network import build_network
from activations import Sigmoid, ReLU, LeakyReLU, Linear, Softmax
from losses import MSE, CrossEntropy
from regularizers import L1, L2
from optimizers import make_optimizer

class NeuralNetRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible neural network regressor.

    Wraps the custom feedforward neural network implementation
    to support scikit-learn methods such as cross-validation
    and hyperparameter selection.
    """

    def __init__(
        self,
        hidden=(50,),
        activation="sigmoid",
        optimizer="adam",
        regularizer="l2",
        gamma=0.03,
        lmbd=0.0,
        epochs=1000,
        batch_size=10,
        seed=67,
    ):
        """
        Initialize the neural network regressor.

        Parameters
        ----------
        hidden : tuple of int
            Number of neurons in each hidden layer.
        activation : str
            Activation function used in the hidden layers.
        optimizer : str
            Optimization algorithm used for training.
        regularizer : str
            Regularization method, either "l1" or "l2".
        gamma : float
            Learning rate for the optimizer.
        lmbd : float
            Regularization strength. Zero disables regularization.
        epochs : int
            Number of training epochs.
        batch_size : int
            Number of samples per mini-batch. Ignored for full-batch GD.
        seed : int
            Random seed used for network initialization.
        """
        self.hidden = hidden
        self.activation = activation
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.gamma = gamma
        self.lmbd = lmbd
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, X, y):
        """
        Build and train the neural network regressor.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Training input data.
        y : array_like of shape (n_samples,)
            Target values for regression.

        Returns
        -------
        self : NeuralNetRegressor
            Fitted estimator.

        Attributes
        ----------
        network_ : object
            Trained neural network.
        history_ : object
            Training history returned by the network.
        """
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)

        # Network architecture
        layers = [X.shape[1], *self.hidden, 1]

        # Hidden activation
        activation_classes = {
            "sigmoid": Sigmoid,
            "relu": ReLU,
        }

        hidden_activation = activation_classes[self.activation]

        activations = (
            [hidden_activation() for _ in self.hidden]
            + [Linear()]
        )

        # Regularization
        if self.lmbd == 0:
            regularizer = None
        elif self.regularizer == "l1":
            regularizer = L1(self.lmbd)
        elif self.regularizer == "l2":
            regularizer = L2(self.lmbd)
        else:
            raise ValueError(f"Unknown regularizer: {self.regularizer}")

        # Build a fresh network for each fit
        self.network_ = build_network(
            layers=layers,
            activations=activations,
            loss=MSE(),
            optimizer=make_optimizer(self.optimizer, self.gamma),
            regularizer=regularizer,
            random_state=self.seed,
        )

        # Train
        batch_size = None if self.optimizer == "gd" else self.batch_size

        self.history_ = self.network_.fit(
            X, y,
            epochs=self.epochs,
            batch_size=batch_size
        )

        return self

    def predict(self, X):
        """
        Predict target values using the trained neural network.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Input data for prediction.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted target values.
        """
        X = np.asarray(X)
        return np.asarray(self.network_.predict(X)).ravel()

