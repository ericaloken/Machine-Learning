# gradients.py

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)  # enable 64-bit precision


def cost_ols(theta, X, y):
    """Compute OLS cost function."""
    return jnp.mean((y - X @ theta) ** 2)


def cost_ridge(theta, X, y, lmbda, n_total=None):
    """Compute Ridge cost function"""
    n = len(y)
    n_pen = n if n_total is None else n_total  # use n_total for mini-batches
    return jnp.mean((y - X @ theta) ** 2) + (lmbda / n_pen) * jnp.sum(theta**2)


def cost_lasso(theta, X, y, lmbda, n_total=None):
    """Compute Lasso cost function"""
    n = len(y)
    n_pen = n if n_total is None else n_total  # use n_total for mini-batches
    return jnp.mean((y - X @ theta) ** 2) + (lmbda / n_pen) * jnp.sum(jnp.abs(theta))


def grad_ols_analytic(theta, X, y):
    """Compute analytical gradient of OLS cost function."""
    n = len(y)
    return (2.0 / n) * X.T @ (X @ theta - y)


def grad_ridge_analytic(theta, X, y, lmbda, n_total=None):
    """ComputeaAnalytical gradient of Ridge cost function."""
    n = len(y)
    n_pen = n if n_total is None else n_total  # use n_total for mini-batches
    return (2.0 / n) * X.T @ (X @ theta - y) + (2.0 * lmbda / n_pen) * theta


def grad_lasso_analytic(theta, X, y, lmbda, n_total=None):
    """Compute analytical gradient of Lasso cost function using subgradient."""
    n = len(y)
    n_pen = n if n_total is None else n_total  # use n_total for mini-batches
    grad = (2.0 / n) * X.T @ (X @ theta - y) + (lmbda / n_pen) * np.sign(
        theta
    )  # for theta_i = 0, the subgradient, we choose 0
    return grad


# jit to compile the functions for speed
# define gradients (once) using automatic differentiation from JAX
grad_ols_ad = jax.jit(jax.grad(cost_ols))
grad_ridge_ad = jax.jit(jax.grad(cost_ridge))
grad_lasso_ad = jax.jit(
    jax.grad(cost_lasso)
)  # JAX returns  +1 for theta_i = 0, which is a valid subgradient
