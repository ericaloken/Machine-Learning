# resampling.py

import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from models import ols_fit, ridge_fit, lasso_fit, RegressionEstimator
from data import design_matrix
from metrics import mse


def bias_variance_bootstrap(
    X_train,
    y_train,
    X_test,
    y_test,
    f_test,
    method="ols",
    lmbda=None,
    n_bootstrap=100,
    bias_reference="y",
    lasso_kwargs=None,
):
    """
    Estimate bias squared, variance, and test MSE using bootstrap resampling.
    The model is repeatedly fitted to a resampled training set and evaluated
    on the same fixed test set. All returned quantties are averaged over test points.

    Parameters
    ----------
    X_train : np.ndarray       Training design matrix.
    y_train : np.ndarray       Training target values.
    X_test : np.ndarray        Test design matrix.
    y_test : np.ndarray        Test target values.
    f_test : np.ndarray        True function values.
    method : str               Regression method, "ols" or "ridge".
    lmbda : float              Ridge regularization parameter.
    n_bootstrap : int          Number of bootstrap samples.
    bias_reference : str       Bias reference, "y" or "f".
    lasso_kwargs : dict or None   Additional keyword arguments for Lasso regression (if method is "lasso").
    Returns
    -------
    tuple                      Bias squared, variance, and test MSE.
    """
    if bias_reference == "f" and f_test is None:
        raise ValueError("f_test is required for bias_reference='f'")
    if bias_reference == "y" and y_test is None:
        raise ValueError("y_test is required for bias_reference='y'")

    n_samples = X_train.shape[0]

    predictions = np.zeros(
        (X_test.shape[0], n_bootstrap)
    )  # rows=test points, columns=bootstrap models

    for i in range(n_bootstrap):
        # Sample training set with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)

        X_boot = X_train[indices]
        y_boot = y_train[indices]

        # Fit new model to the bootstrap training set
        if method == "ols":
            model = ols_fit(X_boot, y_boot)

        elif method == "ridge":
            if lmbda is None:
                raise ValueError("lmbda must be provided for Ridge regression")

            model = ridge_fit(X_boot, y_boot, lmbda)

        elif method == "lasso":
            if lmbda is None:
                raise ValueError("lmbda must be provided for Lasso regression")

            model = lasso_fit(X_boot, y_boot, lmbda, **(lasso_kwargs or {}))
        else:
            raise ValueError("Unknown method. Method must be 'ols', 'ridge' or 'lasso'")

        # Fixed test set so predictions can be compared across models
        predictions[:, i] = model.predict(X_test)

    # Average prediction over bootstrap models (columns) for each test point
    mean_prediction = np.mean(predictions, axis=1)

    # Bias^2
    if bias_reference == "y":
        bias_squared = np.mean((y_test - mean_prediction) ** 2)  # with noise
    else:
        bias_squared = np.mean((f_test - mean_prediction) ** 2)  # without noise

    # Prediction variance across models (averaged over test points)
    variance = np.mean(np.var(predictions, axis=1))

    # Average MSE over all test points and bootstrap models
    mse = np.mean((y_test[:, None] - predictions) ** 2)

    return bias_squared, variance, mse


def bias_variance_degree(
    x_train,
    y_train,
    x_test,
    y_test,
    f_test,
    degrees=range(0, 21),
    method="ols",
    lmbda=None,
    n_bootstrap=100,
    bias_reference="y",
):
    """
    Compute bias, variance, and MSE for different polynomial degrees.
    A design matrix is constructed for each polynomial degree, and the
    bias-variance decomposition is computed using bootstrap resampling.

    Parameters
    ----------
    x_train : np.ndarray       Training input values.
    y_train : np.ndarray       Training target values.
    x_test : np.ndarray        Test input values.
    y_test : np.ndarray        Test target values.
    f_test : np.ndarray        True function values.
    degrees : iterable         Polynomial degrees to evaluate.
    method : str               Regression method, "ols" or "ridge".
    lmbda : float              Ridge regularization parameter.
    n_bootstrap : int          Number of bootstrap samples.
    bias_reference : str       Bias reference, "y" or "f".

    Returns
    -------
    tuple                      Bias, variance, and MSE arrays.
    """

    bias_values = np.zeros(len(degrees))
    variance_values = np.zeros(len(degrees))
    mse_values = np.zeros(len(degrees))

    for i, degree in enumerate(degrees):
        X_train = design_matrix(x_train, degree)
        X_test = design_matrix(x_test, degree)
        bias_values[i], variance_values[i], mse_values[i] = bias_variance_bootstrap(
            X_train,
            y_train,
            X_test,
            y_test,
            f_test,
            method=method,
            lmbda=lmbda,
            n_bootstrap=n_bootstrap,
            bias_reference=bias_reference,
        )

    return bias_values, variance_values, mse_values


def bias_variance_lambda(
    x_train,
    y_train,
    x_test,
    y_test,
    f_test,
    degree=15,
    method="ridge",
    lambdas=np.logspace(-4, 4, 10),
    n_bootstrap=100,
    bias_reference="y",
):
    """
    Compute bias, variance, and MSE for different Ridge parameters.

    Parameters
    ----------
    x_train : np.ndarray       Training input values.
    y_train : np.ndarray       Training target values.
    x_test : np.ndarray        Test input values.
    y_test : np.ndarray        Test target values.
    f_test : np.ndarray        True function values.
    degree : int               Polynomial degree.
    method : str               Regression method.
    lambdas : array-like       Ridge regularization parameters.
    n_bootstrap : int          Number of bootstrap samples.
    bias_reference : str       Bias reference, "y" or "f".

    Returns
    -------
    tuple                      Bias, variance, and MSE arrays.
    """
    bias_values = np.zeros(len(lambdas))
    variance_values = np.zeros(len(lambdas))
    mse_values = np.zeros(len(lambdas))

    for i, lmbda in enumerate(lambdas):
        X_train = design_matrix(x_train, degree)
        X_test = design_matrix(x_test, degree)
        bias_values[i], variance_values[i], mse_values[i] = bias_variance_bootstrap(
            X_train,
            y_train,
            X_test,
            y_test,
            f_test,
            method=method,
            lmbda=lmbda,
            n_bootstrap=n_bootstrap,
            bias_reference=bias_reference,
        )

    return bias_values, variance_values, mse_values


def cross_validation_manual(X, y, method="ols", k=5, random_state=67, lmbda=None):
    """
    Perform manual k-fold cross-validation for a given design matrix and output values.

    Parameters
    ----------
    X : np.ndarray        Design matrix.
    y : np.ndarray        Target values.
    method : str          Regression method, "ols" or "ridge".
    k : int               Number of folds.
    random_state : int    Random seed.
    lmbda : float         Ridge regularization parameter.

    Returns
    -------
    tuple                 Mean and standard deviation of MSE.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    mse_list = []

    for train_idx, test_idx in kf.split(X):
        if method == "ols":
            model = ols_fit(X[train_idx], y[train_idx])
        elif method == "ridge":
            model = ridge_fit(X[train_idx], y[train_idx], lmbda=lmbda)
        else:
            raise ValueError("method must be ols or ridge")

        y_actual = y[test_idx]
        y_pred = model.predict(X[test_idx])
        mse_list.append(mse(y_actual, y_pred))

    return np.mean(mse_list), np.std(mse_list)


def cross_validation_sklearn(
    X, y, method="ols", k=5, random_state=67, lmbda=None, estimator=None
):
    """
    Perform k-fold cross-validation using Scikit-learn for a given design matrix and output values.

    Parameters
    ----------
    X : np.ndarray        Design matrix.
    y : np.ndarray        Target values.
    method : str          Regression method, "ols" or "ridge".
    k : int               Number of folds.
    random_state : int    Random seed.
    lmbda : float         Ridge regularization parameter.
    estimator : sklearn estimator
    Returns
    -------
    tuple                 Mean and standard deviation of MSE.
    """

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    if estimator is None:
        model = RegressionEstimator(method=method, lmbda=lmbda)
    else:
        model = estimator

    scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error")
    mse_scores = -scores

    return np.mean(mse_scores), np.std(mse_scores)


def cross_validation_degree(
    x, y, degrees=range(0, 21), method="ridge", k=5, random_state=67, lmbda=0.01
):
    """
    Perform k-fold cross-validation for different polynomial degrees
    using Sci-kit learn cross_val_score.

    Parameters
    ----------
    x : np.ndarray        Input values.
    y : np.ndarray        Target values.
    degrees : iterable    Polynomial degrees to evaluate.
    method : str          Regression method, "ols" or "ridge".
    k : int               Number of folds.
    random_state : int    Random seed.
    lmbda : float         Ridge regularization parameter.

    Returns
    -------
    tuple                 Mean MSE and standard deviation arrays.
    """
    mse_values = np.zeros(len(degrees))
    std_values = np.zeros(len(degrees))

    for i, degree in enumerate(degrees):
        X = design_matrix(x, degree=degree)
        mse_mean, mse_std = cross_validation_sklearn(
            X, y, method=method, k=k, random_state=random_state, lmbda=lmbda
        )
        mse_values[i] = mse_mean
        std_values[i] = mse_std

    return mse_values, std_values


def cross_validation_lambda(
    x,
    y,
    degree=15,
    method="ridge",
    k=5,
    random_state=67,
    lambdas=np.logspace(-4, 4, 10),
):
    """
    Perform k-fold cross-validation for different Ridge parameters.

    Parameters
    ----------
    x : np.ndarray        Input values.
    y : np.ndarray        Target values.
    degree : int          Polynomial degree.
    method : str          Regression method.
    k : int               Number of folds.
    random_state : int    Random seed.
    lambdas : array-like  Ridge regularization parameters.

    Returns
    -------
    tuple                 Mean MSE and standard deviation arrays.
    """
    mse_values = np.zeros(len(lambdas))
    std_values = np.zeros(len(lambdas))

    for i, lmbda in enumerate(lambdas):
        X = design_matrix(x, degree=degree)
        mse_mean, mse_std = cross_validation_sklearn(
            X, y, method=method, k=k, random_state=random_state, lmbda=lmbda
        )
        mse_values[i] = mse_mean
        std_values[i] = mse_std

    return mse_values, std_values
