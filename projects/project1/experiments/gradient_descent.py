# gradient_descent.py
import numpy as np

from .common import get_gd_data
from metrics import rel_error
from models import Model
from optimizers import gradient_descent, max_learning_rate

# --------------------------------------------------------------
# Task e: Gradient descent for OLS and Ridge
# - compare GD with closed-form OLS and Ridge
# - compare analytical and AD gradients in GD
# - investigate convergence for different learning rates
# - relate convergence to the largest Hessian eigenvalue
# --------------------------------------------------------------


def run_gd_method(cfg: dict, method: str):
    """
    Run gradient descent for OLS or Ridge using analytical and AD gradients.
    The converged parameters are compared with the corresponding
    closed-form solution, and the Hessian is used to determine the
    learning-rate stability limit.

    Parameters
    ----------
    cfg : dict          Configuration dictionary.

    Returns
    -------
    results : dict      Gradient-descent results and Hessian information for OLS/Ridge.
    """
    gd_cfg = cfg["gd"]
    X, y, lmbda = get_gd_data(cfg, method)

    closed_form_model = Model.fit(X, y, method=method, lmbda=lmbda)

    # GD with analytical and with AD gradients
    gd_analytic = Model.fit(
        X,
        y,
        method=method,
        lmbda=lmbda,
        optimizer=gradient_descent,
        use_ad=False,
        gamma_factor=gd_cfg["gamma_factor"],
        max_iter=gd_cfg["max_iter"],
        tol=gd_cfg["tol"],
    )
    gd_ad = Model.fit(
        X,
        y,
        method=method,
        lmbda=lmbda,
        optimizer=gradient_descent,
        use_ad=True,
        gamma_factor=gd_cfg["gamma_factor"],
        max_iter=gd_cfg["max_iter"],
        tol=gd_cfg["tol"],
    )

    # Hessian info
    X_scaled = closed_form_model.scale_X(
        X, closed_form_model.mean_X, closed_form_model.std_X
    )
    gamma_optimal, gamma_max, kappa = max_learning_rate(
        X_scaled, method=method, lmbda=lmbda
    )

    return {
        "theta_closed_form": closed_form_model.theta,
        "theta_analytic_gd": gd_analytic.theta,
        "theta_ad_gd": gd_ad.theta,
        "n_iter_analytic": gd_analytic.n_iter,
        "n_iter_ad": gd_ad.n_iter,
        "error_analytic": rel_error(gd_analytic.theta, closed_form_model.theta),
        "error_ad": rel_error(gd_ad.theta, closed_form_model.theta),
        "gamma_factor": gd_cfg["gamma_factor"],
        "gamma": gd_cfg["gamma_factor"] * gamma_max,
        "gamma_optimal": gamma_optimal,
        "gamma_max": gamma_max,
        "kappa": kappa,
    }


def run_learning_rate_study(cfg: dict, method: str):
    """
    Vary learning rate and perform GD to study convergence, using automatically differentiated
    gradients.
    """
    gd_cfg = cfg["gd"]
    X, y, lmbda = get_gd_data(cfg, method)
    closed_form_model = Model.fit(X, y, method=method, lmbda=lmbda)

    factors = np.asarray(gd_cfg["gamma_factors"])
    iterations = np.zeros(
        len(factors), dtype=int
    )  # number of iterations for each learning rate
    errors = np.zeros(len(factors))

    for i, factor in enumerate(factors):
        gd_model = Model.fit(
            X,
            y,
            method=method,
            lmbda=lmbda,
            optimizer=gradient_descent,
            use_ad=True,
            gamma_factor=factor,
            max_iter=gd_cfg["max_iter"],
            tol=gd_cfg["tol"],
        )
        iterations[i] = gd_model.n_iter
        errors[i] = rel_error(gd_model.theta, closed_form_model.theta)

    return factors, iterations, errors


def run_gd(cfg: dict):
    """
    Run the complete gradient-descent analysis for task e.
    OLS and Ridge are optimized using analytical and automatically
    differentiated gradients and compared with their closed-form
    solutions. Convergence is also investigated for different
    learning rates relative to the Hessian stability limit.
    """
    for method in ("ols", "ridge"):
        results = run_gd_method(cfg, method)

        print(f"\n{method.upper()} gradient descent")
        print("-----------------------------------------------------")
        print(f"gamma_factor                        = {results['gamma_factor']:.2f}")
        print(f"gamma = gamma_factor * gamma_max    = {results['gamma']:.6e}")
        print(f"gamma_optimal                       = {results['gamma_optimal']:.6e}")
        print(f"gamma_max                           = {results['gamma_max']:.6e}")
        print(f"kappa                               = {results['kappa']:.6e}")
        print("\nGradient       Iterations    Relative error")
        print(
            f"Analytical     "
            f"{results['n_iter_analytic']}    "
            f"{results['error_analytic']:.6e}"
        )
        print(
            f"AD             "
            f"{results['n_iter_ad']}    "
            f"{results['error_ad']:.6e}"
        )

        # get iterations and errors for each learning rate in the study
        factors, iterations, errors = run_learning_rate_study(cfg, method)

        print("\nLearning-rate study")
        print("gamma/gamma_max    Iterations    Relative error")

        for factor, n_iter, error in zip(factors, iterations, errors):
            print(f"{factor:15.2f}    " f"{n_iter}    " f"{error:.6e}")
