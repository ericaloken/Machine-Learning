# tests/test_gradient.py

from experiments.common import make_model
from gradients import grad_ols_analytic, grad_ridge_analytic, grad_ols_ad, grad_ridge_ad
from data import design_matrix
import numpy as np

# -------------------------------------------------------------
# Task e: Gradient descent
# - verify that analytical and AD gradients agree up to machine
#   precision or OLS and Ridge
# - compare results from task a) and b) with GD approach
# -------------------------------------------------------------


def verify_gradients(x, y):
    """
    Verify the analytical OLS and Ridge gradients against gradients
    computed using automatic differentiation (AD). A random test scenario
    is created and the maximum absolute difference between the anakytical
    and AD gradient is printed.

    Parameters
    ----------
    x : array-like      Input data.
    y : array-like      Target values.
    """
    rng = np.random.default_rng(67)

    # create test scenario
    degree, lmbda = 5, 1e-2
    X = design_matrix(x, degree)
    theta0 = rng.normal(size=X.shape[1])

    ols_abs_diff = np.abs(grad_ols_ad(theta0, X, y) - grad_ols_analytic(theta0, X, y))
    ridge_abs_diff = np.abs(
        grad_ridge_ad(theta0, X, y, lmbda) - grad_ridge_analytic(theta0, X, y, lmbda)
    )

    print("OLS:    max |AD - analytic| =", np.max(ols_abs_diff))
    print("Ridge:  max |AD - analytic| =", np.max(ridge_abs_diff))


def run_grad_verification(cfg: dict) -> None:
    """
    Run the verification of the CV implemnetations (part of task d).

    Parameters
    ----------
    cfg : dict      Configuration dictionary containing the Ridge regression,
                    data-generation, and plotting settings.
    """
    runge = make_model(cfg)
    verify_gradients(runge.x, runge.y)
