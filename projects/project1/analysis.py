# analysis.py

from data import generate_data, design_matrix, split_data, runge_function
from models import ols_fit, ridge_fit
from resampling import (
    bias_variance_degree,
    cross_validation_manual,
    cross_validation_sklearn,
)
from metrics import mse, r2
import numpy as np


class RungeRegression:
    """
    A class for performing regression analysis on the Runge function.
    """

    def __init__(self, n_points, sigma=0.1, random_state=67, x_sampling="uniform"):
        """
        Initialize the regression analysis and generate the data.

        Parameters
        ----------
        n_points : int        Number of data points.
        sigma : float         Standard deviation of the noise.
        random_state : int    Random seed.
        x_sampling : str      Sampling method for the input values.
        """
        self.n_points = n_points
        self.sigma = sigma
        self.random_state = random_state
        self.x_sampling = x_sampling
        self.x, self.y = generate_data(
            n_points=n_points,
            a=-1,
            b=1,
            sigma=sigma,
            random_state=random_state,
            sampling=x_sampling,
        )

        # set numpy random seed
        np.random.seed(random_state)

    def evaluate(self, X, y, fit_model, test_size=0.2):
        """
        Split the data, fit a model on the training set and evaluate it on the test set.

        Parameters
        ----------
        X : np.ndarray          Design matrix.
        y : np.ndarray          Target values.
        fit_model : callable    Function (X_train, y_train) -> fitted Model.
        test_size : float       Proportion of the dataset to include in the test split.

        Returns
        -------
        tuple                   Test MSE, test R2, and fitted coefficient values.
        """
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=test_size, random_state=self.random_state
        )
        model = fit_model(X_train, y_train)
        y_pred = model.predict(X_test)
        MSE = mse(y_test, y_pred)
        R2 = r2(y_test, y_pred)
        thetas = model.theta.copy()

        return MSE, R2, thetas

    def split_design_data(self, degree, test_size=0.2):
        """
        Split the design matrix and target values into training and test sets.

        Parameters
        ----------
        degree : int          Polynomial degree.
        test_size : float     Proportion of the dataset to include in the test split.

        Returns
        -------
        tuple                 Training and test sets for design matrix and target values.
        """
        X = design_matrix(self.x, degree=degree)
        X_train, X_test, y_train, y_test = split_data(
            X, self.y, test_size=test_size, random_state=self.random_state
        )
        return X_train, X_test, y_train, y_test

    def evaluate_ols_multiple_degrees(self, degrees=range(1, 16), test_size=0.2):
        """
        Evaluate OLS for different polynomial degrees.

        Parameters
        ----------
        degrees : iterable    Polynomial degrees to evaluate.

        Returns
        -------
        tuple                 MSE, R2, and fitted coefficient values.
        """
        mse_values = np.zeros(len(degrees))
        r2_values = np.zeros(len(degrees))
        parameter_values = []

        for i, degree in enumerate(degrees):
            X = design_matrix(self.x, degree=degree)
            mse_values[i], r2_values[i], parameters = self.evaluate(
                X, self.y, fit_model=ols_fit, test_size=test_size
            )
            parameter_values.append(parameters)

        return mse_values, r2_values, parameter_values

    def evaluate_ridge_multiple_degrees(
        self, degrees=range(1, 16), lmbda=0.01, test_size=0.2
    ):
        """
        Evaluate Ridge regression for different polynomial degrees.

        Parameters
        ----------
        degrees : iterable    Polynomial degrees to evaluate.
        lmbda : float         Ridge regularization parameter.
        test_size : float     Proportion of the dataset to include in the test split.

        Returns
        -------
        tuple                 MSE, R2, and fitted coefficient values.
        """
        mse_values = np.zeros(len(degrees))
        r2_values = np.zeros(len(degrees))
        parameter_values = []

        for i, degree in enumerate(degrees):
            X = design_matrix(self.x, degree=degree)

            def ridge_fit_fixed_lambda(X_train, y_train):
                return ridge_fit(X_train, y_train, lmbda=lmbda)

            mse_values[i], r2_values[i], parameters = self.evaluate(
                X,
                self.y,
                fit_model=ridge_fit_fixed_lambda,
                test_size=test_size,
            )

            parameter_values.append(parameters)

        return mse_values, r2_values, parameter_values

    def evaluate_ridge_multiple_lambdas(
        self, degree=15, lambdas=np.logspace(-4, 4, 10), test_size=0.2
    ):
        """
        Evaluate Ridge regression for different lambda values.

        Parameters
        ----------
        degree : int          Polynomial degree.
        lambdas : array-like  Regularization parameters to evaluate.

        Returns
        -------
        tuple                 MSE, R2, and fitted coefficient values.
        """
        mse_values = np.zeros(len(lambdas))
        r2_values = np.zeros(len(lambdas))
        parameter_values = []

        X = design_matrix(self.x, degree=degree)

        for i, lmbda in enumerate(lambdas):

            def ridge_fit_w_lmbda(X_train, y_train):
                # fixes the lambda value for ridge_fit
                return ridge_fit(X_train, y_train, lmbda=lmbda)

            mse_values[i], r2_values[i], parameters = self.evaluate(
                X, self.y, fit_model=ridge_fit_w_lmbda, test_size=test_size
            )
            parameter_values.append(parameters)

        return mse_values, r2_values, parameter_values

    def train_test_mse_vs_degree(self, degrees=range(0, 31), test_size=0.2):
        """
        Compute training and test MSE for different polynomial degrees.

        Parameters
        ----------
        degrees : iterable    Polynomial degrees to evaluate.
        test_size : float     Proportion of the dataset to include in the test split.

        Returns
        -------
        tuple                 Training and test MSE arrays.
        """
        train_mse = np.zeros(len(degrees))
        test_mse = np.zeros(len(degrees))

        for i, degree in enumerate(degrees):
            X = design_matrix(self.x, degree=degree)
            X_train, X_test, y_train, y_test = split_data(
                X, self.y, test_size=test_size, random_state=self.random_state
            )

            model = ols_fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_mse[i] = mse(y_train, y_train_pred)
            test_mse[i] = mse(y_test, y_test_pred)

        return train_mse, test_mse

    def bias_variance_bootstrap(
        self,
        degrees=range(0, 31),
        n_bootstrap=100,
        bias_reference="y",
        method="ols",
        lmbda=None,
        test_size=0.2,
    ):
        """
        Compute bias, variance, and test error using bootstrap resampling.

        Parameters
        ----------
        degrees : iterable      Polynomial degrees to evaluate.
        n_bootstrap : int       Number of bootstrap samples.
        bias_reference : str    Reference used to compute the bias.
        method : str            Regression method.
        lmbda : float or None   Ridge regularization parameter.
        test_size : float       Proportion of the dataset to include in the test split.

        Returns
        -------
        tuple                   Bias, variance, and test MSE arrays.
        """
        x_train, x_test, y_train, y_test = split_data(
            self.x, self.y, test_size=test_size, random_state=self.random_state
        )

        bias_values, variance_values, mse_values = bias_variance_degree(
            x_train,
            y_train,
            x_test,
            y_test,
            f_test=runge_function(x_test),
            degrees=degrees,
            method=method,
            lmbda=lmbda,
            n_bootstrap=n_bootstrap,
            bias_reference=bias_reference,
        )

        return bias_values, variance_values, mse_values


def print_cv_implementations_comparison(
    x,
    y,
    degree=10,
    method="ols",
    k=5,
    random_state=67,
    lmbda=0.01,
) -> None:
    """
    Compare manual and Scikit-learn cross-validation results.

    Parameters
    ----------
    x : np.ndarray        Input values.
    y : np.ndarray        Target values.
    degree : int          Polynomial degree.
    method : str          Regression method.
    k : int               Number of folds.
    random_state : int    Random seed.
    lmbda : float         Ridge regularization parameter.
    """
    X = design_matrix(x, degree=degree)

    # MSE form our own implentation
    mse_own, std_own = cross_validation_manual(
        X, y, method=method, k=k, random_state=random_state, lmbda=lmbda
    )

    # MSE from Scikit-learn implementation
    mse_sklearn, std_sklearn = cross_validation_sklearn(
        X, y, method=method, k=k, random_state=random_state, lmbda=lmbda
    )

    print("-----------------------------------------------------------")
    print(f"CV comparison: {method}, k={k}")
    print(f"{'Implementation:':<18} {'MSE:':>14} {'Std:':>14}")
    print(f"{'Own':<18} {mse_own:>14.8f} {std_own:>14.8f}")
    print(f"{'Scikit-learn':<18} {mse_sklearn:>14.8f} {std_sklearn:>14.8f}")
    print(
        f"{'Difference':<18} "
        f"{mse_own - mse_sklearn:>14.3e} "
        f"{std_own - std_sklearn:>14.3e}\n"
    )
