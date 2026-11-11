# linear_models.py

# copy of models.py with removal of gradient descent and other optimizers, since they are now in optimizers.py

import numpy as np
from dataclasses import dataclass
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning
import warnings


@dataclass
class Model:
    theta: np.ndarray
    mean_X: np.ndarray
    std_X: np.ndarray
    mean_y: float
    std_y: float
    n_iter: int | None = None  # iterations for optimizer (none for closed form)

    """
    Dataclass representing a fitted regression model. Fitted regression coefficients and
    scaling parameters are computed using the training data.

    Attributes
    ----------
    theta : np.ndarray    Model coefficients.
    mean_X : np.ndarray   Training feature means.
    std_X : np.ndarray    Training feature standard deviations.
    mean_y : float        Training target mean.
    std_y : float         Training target standard deviation.
    """

    def predict(self, X):
        """
        Predict target values for input data.

        Parameters
        ----------
        X : np.ndarray    Input design matrix.

        Returns
        -------
        np.ndarray        Predicted target values.
        """
        X_scaled = self.scale_X(
            X, self.mean_X, self.std_X
        )  # scale each column (except intercept)
        y_pred_scaled = X_scaled @ self.theta
        y_pred = y_pred_scaled * self.std_y + self.mean_y  # scale back to original
        return y_pred

    @staticmethod  # staticmethod suggested in debugging by claude sonnet 5
    def scale_X(X, mean_X, std_X):
        """
        Scale features using training-set statistics. Used before model exists.

        Parameters
        ----------
        X : np.ndarray         Input design matrix.
        mean_X : np.ndarray    Training feature means.
        std_X : np.ndarray     Training feature standard deviations.

        Returns
        -------
        np.ndarray             Scaled design matrix.
        """
        X_scaled = np.asarray(X, dtype=float).copy()
        X_scaled[:, 1:] = (X_scaled[:, 1:] - mean_X) / std_X  # skip bias column
        return X_scaled

    @staticmethod
    def training_statistics(X_train, y_train):
        """
        Scaling parameters from the training data. The bias column is skipped,
        since it has zero standard deviation.

        Returns
        -------
        mean_X, std_X : np.ndarray    Feature means and standard deviations.
        mean_y, std_y : float         Target mean and standard deviation.
        """
        mean_X = np.mean(X_train[:, 1:], axis=0)  # skip bias column
        std_X = np.std(X_train[:, 1:], axis=0)
        mean_y = np.mean(y_train)
        std_y = np.std(y_train)
        return mean_X, std_X, mean_y, std_y

    def scale(self, X, y):
        """
        Scale X and y with this model's training statistics (the same scaling as in fit).
        Used after the model is fitted, e.g. for test data.

        Parameters
        ----------
        X : np.ndarray        Design matrix.
        y : np.ndarray        Target values.

        Returns
        -------
        X_scaled : np.ndarray
        y_scaled : np.ndarray
        """
        X_scaled = self.scale_X(X, self.mean_X, self.std_X)
        y_scaled = (y - self.mean_y) / self.std_y
        return X_scaled, y_scaled

    @classmethod
    def fit(
        cls,
        X_train,
        y_train,
        method="ols",
        lmbda=None,
    ):
        """
        Scale training data and fit an OLS, Ridge, or Lasso model.
        OLS and Ridge are fitted using closed-form solutions.
        The fitted model is returned as a Model instance.

        Parameters
        ----------
        X_train : np.ndarray    Training design matrix.
        y_train : np.ndarray    Training target values.
        method : str            Regression method, "ols", "ridge" or "lasso".
        lmbda : float           Ridge/Lasso regularization parameter.

        Returns
        -------
        Model                   Fitted model instance.
        """

        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)

        # scaling parameters from training data
        mean_X, std_X, mean_y, std_y = cls.training_statistics(X_train, y_train)

        # scale X and y
        X_scaled = cls.scale_X(X_train, mean_X, std_X)
        y_scaled = (y_train - mean_y) / std_y

        # closed-form solutions for OLS and Ridge
        theta = closed_form_solutions(X_scaled, y_scaled, method=method, lmbda=lmbda)

        return cls(theta, mean_X, std_X, mean_y, std_y, n_iter=None)


class RegressionEstimator(BaseEstimator, RegressorMixin):
    """Scikit-learn wrapper for the custom regression models."""

    def __init__(self, method="ols", lmbda=None, optimizer=None, optimizer_kwargs=None):
        self.method = method
        self.lmbda = lmbda
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

    def fit(self, X, y):
        if self.optimizer_kwargs is None:
            kwargs = {}
        else:
            kwargs = self.optimizer_kwargs

        self.model_ = Model.fit(X, y, method=self.method, lmbda=self.lmbda, **kwargs)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def closed_form_solutions(X, y, method="ols", lmbda=None):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)  # svd to compute inverse of X

    if method == "ols":
        # pseudo-inverse: ignore singular values below a relative tolerance (same as np.linalg.pinv) suggested by claude sonnet 5
        eps = np.finfo(float).eps  # machine precision
        s_max = s[0]  # largest singular value
        rcond = max(X.shape) * eps  # threshold for small singular values
        s_inv = np.where(
            s > rcond * s_max, 1.0 / s, 0.0
        )  # invert singular values above threshold, else 0
        return Vt.T @ (s_inv * (U.T @ y))  # theta

    if method == "ridge":
        # ridge regression
        if lmbda is None:
            raise ValueError("lmbda must be provided")
        factor = s / (s**2 + lmbda)
        return Vt.T @ (factor * (U.T @ y))  # theta

    raise ValueError(
        "closed form method must be 'ols' or 'ridge'"
    )  # if ridge or ols not chosen


# --------------------------------
#       Wrapper functions
# --------------------------------


def ols_fit(X_train, y_train, **kwargs):
    """
    Fit an OLS model using the Model class.

    Parameters
    ----------
    X_train : np.ndarray    Training design matrix.
    y_train : np.ndarray    Training target values.

    Returns
    -------
    Model                   Fitted model instance.
    """
    return Model.fit(X_train, y_train, method="ols", **kwargs)


def ridge_fit(X_train, y_train, lmbda, **kwargs):
    """
    Fit a Ridge regression model using the Model class.

    Parameters
    ----------
    X_train : np.ndarray    Training design matrix.
    y_train : np.ndarray    Training target values.
    lmbda : float           Ridge regularization parameter.

    Returns
    -------
    Model                   Fitted model instance.
    """
    return Model.fit(X_train, y_train, method="ridge", lmbda=lmbda, **kwargs)


def sklearn_lasso_fit(X_train, y_train, lmbda):
    """
    Fit a Lasso regression model using Scikit-Learn's Lasso implementation.
    From project1.
    """
    mean_X, std_X, mean_y, std_y = Model.training_statistics(X_train, y_train)
    model = Model(np.zeros(X_train.shape[1]), mean_X, std_X, mean_y, std_y)

    X_scaled, y_scaled = model.scale(X_train, y_train)

    alpha = lmbda / (2 * len(y_train))  # Scikit-Learn's alpha parameter convention

    # data is centered so we can set fit_intercept=False
    lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=100000, tol=1e-10)

    # coordinate descent does not converge for high degree and small lambda
    # (ill-conditioned design matrix); this is discussed in the report, so the
    # warning is silenced to keep the output readable
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", ConvergenceWarning)
    #     lasso.fit(X_scaled[:, 1:], y_scaled)  # skip the bias term in X_scaled
    lasso.fit(X_scaled[:, 1:], y_scaled)
    print(f"Iterations: {lasso.n_iter_}")
    print(f"Dual gap: {lasso.dual_gap_:.3e}")

    theta = np.concatenate(([0.0], lasso.coef_))  # 0 for the bias term

    return Model(theta=theta, mean_X=mean_X, std_X=std_X, mean_y=mean_y, std_y=std_y)
