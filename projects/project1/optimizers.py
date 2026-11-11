# optimizers.py

import numpy as np


def general_optimizer(
    gradient,
    theta0,
    X,
    y,
    alpha,
    beta1,
    beta2,
    epsilon,
    a,
    b,
    batch_size=None,
    max_iter=10000,
    tol=1e-8,
    random_state=None,
    recorder_func=None,
    n_epochs=None,
    schedule=None,
    **kwargs,
):
    """
    Optimize model parameters using a generalized gradient-based method.
    The generalized update can reproduce gradient descent, stochastic
    gradient descent, momentum, AdaGrad, RMSProp, and Adam through
    different choices of the optimizer parameters.

    Note that X and y are assumed to be standardized

    Parameters
    ----------
    gradient : callable                     Function that computes the gradient of the cost function with
                                            respect to the model parameters.
    theta0 : array-like                     Initial parameter values (guess).
    X : ndarray                             Standardized design matrix.
    y : ndarray                             Standardized target values.
    alpha : float                           Learning rate.
    beta1 : float, optional                 Decay parameter for the first moment estimate.
    beta2 : float, optional                 Decay parameter for the second moment estimate.
    epsilon : float, optional               Small constant added to the denominator for numerical stability.
    a : int, optional                       Switch controlling accumulation of squared gradients.
    b : int, optional                       Switch controlling bias correction.
    batch_size : int or None, optional      Number of samples used to compute each gradient.
                                            If None, the full data set is used.
    max_iter : int, optional                Maximum number of optimization iterations.
    tol : float, optional                   Convergence tolerance based on the change in the parameter vector.
    random_state : int or None, optional    Random seed used when drawing mini-batches.
    recorder_func : callable, optional      Function to record the optimization progress.
    n_epochs : int, optional                Number of epochs to train for. ie. number of passes through the entire dataset. If provided, overrides max_iter.
    schedule : callable, optional           Function to schedule the learning rate.
    **kwargs                                Additional keyword arguments passed to the gradient function,
                                            such as the regularization parameter for Ridge regression.

    Returns
    -------
    theta : ndarray                         Optimized model parameters.
    n_iter : int                            Number of iterations performed before convergence or reaching max_iter.
    """
    # initialize
    theta = np.asarray(theta0, dtype=float).copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    n = len(y)

    rng = np.random.default_rng(random_state)

    # use full data set batch_size is not provided
    if batch_size == None:
        batch_size = n
    n_batches = int(np.ceil(n / batch_size))  # mini-batches per epoch

    if n_epochs is not None:
        max_iter = n_epochs * n_batches  # total number of iterations

    for t in range(1, max_iter + 1):
        # use full batch or mini-batch gradient
        if batch_size == n:
            X_batch = X
            y_batch = y
        else:
            # next mini-batch
            batch = (t - 1) % n_batches  # position in epoch
            if batch == 0:
                order = rng.permutation(
                    n
                )  # shuffle the data at the start of each epoch
            indices = order[batch * batch_size : (batch + 1) * batch_size]
            X_batch = X[indices]
            y_batch = y[indices]

        # calculate gradient (OlS or Ridge, depending on if lmbda is provided)
        grad = gradient(theta, X_batch, y_batch, **kwargs)

        # first moment
        m = beta1 * m + (1.0 - beta1) * grad

        # second moment
        v = beta2 * v + (1.0 - beta2 + a) * grad**2

        # bias correction if is b = 1
        m_hat = m / (1.0 - b * beta1**t)
        v_hat = v / (1.0 - b * beta2**t)

        # learning rate schedule
        if schedule is not None:
            alpha_t = alpha * schedule(t)
        else:
            alpha_t = alpha

        # update parameters
        theta_new = theta - alpha_t * m_hat / (np.sqrt(v_hat) + epsilon)

        if recorder_func is not None:
            recorder_func(theta_new)

        converged = np.linalg.norm(theta_new - theta) < tol

        theta = theta_new  # checks convergence before updating theta and breaks after update if converged
        if converged:
            break

    return theta, int(t)


# Wrapper functions for specific optimization algorithms


def gradient_descent(
    gradient,
    theta0,
    X,
    y,
    gamma,
    max_iter=10000,
    tol=1e-8,
    **kwargs,
):
    """
    Optmize model parameteres using plain gradient descent.

    Returns
    -------
    theta : ndarray         Optimized model parameters.
    n_iter : int            Number of iterations performed.
    """
    return general_optimizer(
        gradient=gradient,
        theta0=theta0,
        X=X,
        y=y,
        alpha=gamma,
        beta1=0.0,
        beta2=1.0,
        epsilon=1.0,
        a=0.0,
        b=0.0,
        batch_size=len(y),
        max_iter=max_iter,
        tol=tol,
        **kwargs,
    )


def stochastic_gradient_descent(
    gradient,
    theta0,
    X,
    y,
    gamma,
    batch_size,
    max_iter=10000,
    tol=1e-8,
    **kwargs,
):
    """
    Optmize model parameteres using stochastic gradient descent with mini-batches.

    Returns
    -------
    theta : ndarray         Optimized model parameters.
    n_iter : int            Number of iterations performed.
    """
    return general_optimizer(
        gradient=gradient,
        theta0=theta0,
        X=X,
        y=y,
        alpha=gamma,
        beta1=0.0,
        beta2=1.0,
        epsilon=1.0,
        a=0.0,
        b=0.0,
        batch_size=batch_size,
        max_iter=max_iter,
        tol=tol,
        **kwargs,
    )


def momentum(
    gradient,
    theta0,
    X,
    y,
    gamma,
    beta,
    batch_size,
    max_iter=10000,
    tol=1e-8,
    **kwargs,
):
    """
    Optmize model parameteres using gradient descent with momentum.

    Returns
    -------
    theta : ndarray         Optimized model parameters.
    n_iter : int            Number of iterations performed.
    """
    alpha = gamma / (1 - beta)
    return general_optimizer(
        gradient=gradient,
        theta0=theta0,
        X=X,
        y=y,
        alpha=alpha,
        beta1=beta,
        beta2=1.0,
        epsilon=1.0,
        a=0.0,
        b=0.0,
        batch_size=batch_size,
        max_iter=max_iter,
        tol=tol,
        **kwargs,
    )


def adagrad(
    gradient,
    theta0,
    X,
    y,
    gamma,
    epsilon0,
    batch_size,
    max_iter=10000,
    tol=1e-8,
    **kwargs,
):
    """
    Optmize model parameteres using gradient descent using AdaGrad.
    AdaGrad adapts the learning rate for each parameter by accumulating
    the squared gradients over previous iterations.

    Returns
    -------
    theta : ndarray         Optimized model parameters.
    n_iter : int            Number of iterations performed.
    """
    return general_optimizer(
        gradient=gradient,
        theta0=theta0,
        X=X,
        y=y,
        alpha=gamma,
        beta1=0.0,
        beta2=1.0,
        epsilon=epsilon0,
        a=1.0,
        b=0.0,
        batch_size=batch_size,
        max_iter=max_iter,
        tol=tol,
        **kwargs,
    )


def rmsprop(
    gradient,
    theta0,
    X,
    y,
    gamma,
    epsilon0,
    rho,
    batch_size,
    max_iter=10000,
    tol=1e-8,
    **kwargs,
):
    """
    Optmize model parameteres using gradient descent using RMSProp.
    RMSProp adapts the learning rate for each parameter using an
    exponentially weighted moving average of the squared gradients.

    Returns
    -------
    theta : ndarray         Optimized model parameters.
    n_iter : int            Number of iterations performed.
    """
    return general_optimizer(
        gradient=gradient,
        theta0=theta0,
        X=X,
        y=y,
        alpha=gamma,
        beta1=0.0,
        beta2=rho,
        epsilon=epsilon0,
        a=0.0,
        b=0.0,
        batch_size=batch_size,
        max_iter=max_iter,
        tol=tol,
        **kwargs,
    )


def adam(
    gradient,
    theta0,
    X,
    y,
    alpha,
    epsilon0,
    beta1,
    beta2,
    batch_size,
    max_iter=10000,
    tol=1e-8,
    **kwargs,
):
    """
    Optmize model parameteres using gradient descent using Adam.
    Adam combines a first-moment estimate of the gradient with an
    exponentially weighted second-moment estimate and applies bias
    correction to both.

    Returns
    -------
    theta : ndarray         Optimized model parameters.
    n_iter : int            Number of iterations performed.
    """
    return general_optimizer(
        gradient=gradient,
        theta0=theta0,
        X=X,
        y=y,
        alpha=alpha,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon0,
        a=0.0,
        b=1.0,
        batch_size=batch_size,
        max_iter=max_iter,
        tol=tol,
        **kwargs,
    )


def max_learning_rate(X, method="ols", lmbda=0.0):
    """
    Compute the theoretical upper bound of the learning rate for gradient descent.

    Parameters
    ----------
    X : ndarray                             Design matrix.
    method : {"ols", "ridge"}, optional     Regression method.
    lmbda : float, optional                 Ridge regularization parameter.

    Returns
    -------
    gamma_optimal: float                    Optimal learning rate.
    gamma_max : float                       Upper learning rate bound.
    kappa: float                            Condition number.
    """
    n = X.shape[0]

    H = (
        (2.0 / n) * X.T @ X
    )  # Hessian for OLS and Lasso (Lasso has the same Hessian as OLS, but with a non-differentiable term, but we need a learning rate calculation.)
    if method == "ridge":  # Hessian for Ridge
        H = H + 2.0 * (lmbda / n) * np.eye(X.shape[1])
    elif method not in ("ols", "ridge", "lasso"):
        raise ValueError("Method must be ols, ridge, or lasso.")

    eigenvaulues = np.linalg.eigvalsh(H)
    min_eigenval = eigenvaulues.min()
    max_eigenval = eigenvaulues.max()

    gamma_optimal = 2.0 / (max_eigenval + min_eigenval)
    gamma_max = 2.0 / max_eigenval
    kappa = max_eigenval / min_eigenval

    return gamma_optimal, gamma_max, kappa
