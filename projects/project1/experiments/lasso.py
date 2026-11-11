# lasso.py

# ------------------------------
# Task g: Lasso regression
# 1. Non diff theta at zero
# 2. Lasso over lambda sweep for fixed deg. Test MSE, R2, and coeffs, own GD vs Scikit-Learn's Lasso.
# 3. Optimizers from part_e for Lasso.
# 4. OLS, Lasso, Ridge comparison for different optimizers and learning rates.
# ------------------------------


import numpy as np
import jax
import jax.numpy as jnp

from .common import (
    make_model,
    logspace,
    OPTIMIZER_NAMES,
    optimizer_settings,
    iterations_for_target,
    sklearn_lasso_fit,
)
from metrics import mse, r2, rel_error
from models import lasso_fit, ols_fit, ridge_fit
from optimizers import gradient_descent, max_learning_rate
from gradients import cost_lasso, grad_lasso_analytic, grad_lasso_ad
import plotting as plot

# ------------------------------
# helpers
# ------------------------------


def get_lasso_data(cfg: dict):
    """
    Generate data for Lasso regression.
    """
    runge = make_model(cfg)
    return runge.split_design_data(
        degree=cfg["lasso"]["degree"], test_size=cfg["data"]["test_size"]
    )


# 1. run subgradient check


def run_ad_subgradient_check(cfg: dict) -> dict:
    """
    Check  what JAX returns for the subgradient of the Lasso cost function, and compare with our own implementation.

    Parameters:
    ----------
    cfg : dict
        Configuration dictionary containing parameters for Lasso regression.

    Returns:
    -------
    dict
        A dictionary containing the results of the subgradient check, including:
        - d_abs_at_zero: The subgradient of the absolute value function at 0.
        - diff_at_zero: The maximum absolute difference between the analytic and automatic differentiation gradients at theta = 0.
        - diff_random: The maximum absolute difference between the analytic and automatic differentiation gradients at a random theta.
        - lmbda_over_n: The regularization parameter divided by the number of samples.
        - error_analytic: The relative error between the analytic and reference solutions.
        - error_ad: The relative error between the automatic differentiation and reference solutions.
        - diff_analytic_ad: The relative error between the analytic and automatic differentiation solutions.
    """
    lasso_cfg = cfg["lasso"]
    lmbda = lasso_cfg["lmbda"]
    X_train, _, y_train, _ = get_lasso_data(cfg)  # not using test data

    reference = sklearn_lasso_fit(X_train, y_train, lmbda)
    X_scaled, y_scaled = reference.scale(X_train, y_train)

    # gradients at theta = 0
    theta_zero = np.zeros(X_scaled.shape[1])
    rng = np.random.default_rng(seed=cfg["data"]["seed"])
    theta_random = rng.normal(size=X_scaled.shape[1])

    # gradients at theta = 0 and random theta
    diff_at_zero = np.max(
        np.abs(
            grad_lasso_analytic(theta_zero, X_scaled, y_scaled, lmbda)
            - grad_lasso_ad(theta_zero, X_scaled, y_scaled, lmbda)
        )
    )
    diff_random = np.max(
        np.abs(
            grad_lasso_analytic(theta_random, X_scaled, y_scaled, lmbda)
            - grad_lasso_ad(theta_random, X_scaled, y_scaled, lmbda)
        )
    )

    # plain GD with both gradients
    gd_analytic = lasso_fit(
        X_train,
        y_train,
        lmbda,
        optimizer=gradient_descent,
        use_ad=False,
        gamma_factor=lasso_cfg["gamma_factor"],
        max_iter=lasso_cfg["max_iter"],
        tol=lasso_cfg["tol"],
    )
    gd_ad = lasso_fit(
        X_train,
        y_train,
        lmbda,
        optimizer=gradient_descent,
        use_ad=True,
        gamma_factor=lasso_cfg["gamma_factor"],
        max_iter=lasso_cfg["max_iter"],
        tol=lasso_cfg["tol"],
    )

    return {
        "d_abs_at_zero": float(jax.grad(jnp.abs)(0.0)),
        "diff_at_zero": diff_at_zero,
        "diff_random": diff_random,
        "lmbda_over_n": lmbda / len(y_train),
        "error_analytic": rel_error(gd_analytic.theta, reference.theta),
        "error_ad": rel_error(gd_ad.theta, reference.theta),
        "diff_analytic_ad": rel_error(gd_ad.theta, gd_analytic.theta),
    }


# 2. run lasso over lambda sweep


def run_lasso_over_lambda(cfg: dict):
    """
    Evaluate Lasso (own GD and Scikit-Learn) over lambda at fixed degree.

    Returns
    -------
    lambda_vals : np.ndarray      Lambda values (our convention).
    mse_vals : np.ndarray         Test MSE, shape (2, n_lambdas): row 0 own GD, row 1 Scikit-Learn.
    r2_vals : np.ndarray          Test R^2, same shape.
    params_own : list             Own GD coefficients for each lambda.
    params_sklearn : list         Scikit-Learn coefficients for each lambda.
    """
    lasso_cfg = cfg["lasso"]
    lambda_vals = logspace(lasso_cfg["lambda_test_log"])

    X_train, X_test, y_train, y_test = get_lasso_data(cfg)

    mse_vals = np.zeros((2, len(lambda_vals)))
    r2_vals = np.zeros((2, len(lambda_vals)))
    params_own = []
    params_sklearn = []

    for i, lmbda in enumerate(lambda_vals):
        own_model = lasso_fit(
            X_train,
            y_train,
            lmbda,
            gamma_factor=lasso_cfg["gamma_factor"],
            max_iter=lasso_cfg["max_iter"],
            tol=lasso_cfg["tol"],
        )
        sklearn_model = sklearn_lasso_fit(X_train, y_train, lmbda)

        for j, model in enumerate((own_model, sklearn_model)):
            y_pred = model.predict(X_test)
            mse_vals[j, i] = mse(y_test, y_pred)
            r2_vals[j, i] = r2(y_test, y_pred)

        params_own.append(own_model.theta)
        params_sklearn.append(sklearn_model.theta)

    return lambda_vals, mse_vals, r2_vals, params_own, params_sklearn


# 3. optimizers from part_e


def lasso_error_vs_sklearn(cfg: dict, optimizer_name: str, gamma: float):
    """
    Compare our own Lasso implementation for one optimizer with Scikit-Learn's Lasso implementation.


    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing parameters for Lasso regression.
    optimizer_name : str
        Name of the optimizer to use (e.g., "gradient_descent", "momentum", "adagrad", "rmsprop", "adam").
    gamma : float
        Learning rate for the optimizer.
    """
    lasso_cfg = cfg["lasso"]
    lmbda = lasso_cfg["lmbda"]
    X_train, _, y_train, _ = get_lasso_data(cfg)

    # ref solution and its cost
    reference = sklearn_lasso_fit(X_train, y_train, lmbda)
    X_scaled, y_scaled = reference.scale(X_train, y_train)
    cost_ref = cost_lasso(reference.theta, X_scaled, y_scaled, lmbda)

    errors = []

    def record_error(theta):
        errors.append(rel_error(theta, reference.theta))

    # optimizer settings from f) ([adaptive_gd])
    optimizer, optimizer_params = optimizer_settings(optimizer_name, gamma, cfg)

    with np.errstate(
        over="ignore", invalid="ignore"
    ):  # ignore warnings from diverging runs (too large learning rate)
        model = lasso_fit(
            X_train,
            y_train,
            lmbda,
            optimizer=optimizer,
            max_iter=lasso_cfg["max_iter"],
            tol=lasso_cfg["tol"],
            recorder_func=record_error,
            **optimizer_params,
        )
    cost_gap = cost_lasso(model.theta, X_scaled, y_scaled, lmbda) - cost_ref

    return np.array(errors), cost_gap


def run_lasso_optimizers(cfg: dict):
    """
    Run different optimizers for Lasso regression and compare their performance.
    """
    lasso_cfg = cfg["lasso"]
    gammas = np.asanyarray(lasso_cfg["learning_rates"])
    target = lasso_cfg["target_error"]

    shape = (len(OPTIMIZER_NAMES), len(gammas))
    iterations = np.zeros(shape)
    final_errors = np.zeros(shape)
    cost_gaps = np.zeros(shape)

    for i, name in enumerate(OPTIMIZER_NAMES):
        for j, gamma in enumerate(gammas):
            errors, cost_gap = lasso_error_vs_sklearn(cfg, name, gamma)
            iterations[i, j] = iterations_for_target(errors, target)
            final_errors[i, j] = errors[-1]
            cost_gaps[i, j] = cost_gap

    return gammas, iterations, final_errors, cost_gaps


def best_learning_rates(gammas, final_errors):
    """
    Determine the best learning rate for each optimizer.
    """
    best = np.zeros(len(OPTIMIZER_NAMES))
    for i in range(len(OPTIMIZER_NAMES)):
        # if all final errors are NaN, no learning rate worked
        if np.all(np.isnan(final_errors[i])):
            best[i] = np.nan
        else:
            best[i] = gammas[np.nanargmin(final_errors[i])]
    return best


# 4. OLS, Lasso, Ridge comparison for different optimizers and learning rates


def run_method_comparison(cfg: dict):
    """
    Compare OLS, Lasso, and Ridge regression methods for different optimizers and learning rates.
    Prints were made with Claude Sonnet 5 for good readability.
    """
    lasso_cfg = cfg["lasso"]
    lmbda = lasso_cfg["lmbda"]
    X_train, X_test, y_train, y_test = get_lasso_data(cfg)
    gd_settings = dict(
        optimizer=gradient_descent,
        gamma_factor=lasso_cfg["gamma_factor"],
        max_iter=lasso_cfg["max_iter"],
        tol=lasso_cfg["tol"],
    )

    models = [
        ("OLS (closed form)", ols_fit(X_train, y_train)),
        ("OLS (GD)", ols_fit(X_train, y_train, **gd_settings)),
        ("Ridge (closed form)", ridge_fit(X_train, y_train, lmbda)),
        ("Ridge (GD)", ridge_fit(X_train, y_train, lmbda, **gd_settings)),
        ("Lasso (Scikit-Learn)", sklearn_lasso_fit(X_train, y_train, lmbda)),
        ("Lasso (GD)", lasso_fit(X_train, y_train, lmbda, **gd_settings)),
    ]

    results = []
    for label, model in models:
        results.append((label, mse(y_test, model.predict(X_test)), model.theta))
    return results


# run everything


def run_lasso(cfg: dict):
    lasso_cfg = cfg["lasso"]
    degree = lasso_cfg["degree"]
    lmbda = lasso_cfg["lmbda"]
    max_iter = lasso_cfg["max_iter"]
    target = lasso_cfg["target_error"]

    # 1. the non-differentiable point
    check = run_ad_subgradient_check(cfg)
    print("\nLasso: the non-differentiable point |theta| = 0")
    print("-----------------------------------------------------")
    print(f"jax.grad(jnp.abs)(0.0)                   = {check['d_abs_at_zero']}")
    print(
        f"max|analytic - AD| gradient at theta = 0 = {check['diff_at_zero']:.3e}"
        f"   (lmbda/n = {check['lmbda_over_n']:.3e})"
    )
    print(f"max|analytic - AD| gradient at random    = {check['diff_random']:.3e}")
    print(
        f"GD, analytic subgradient: rel. error to Scikit-Learn = {check['error_analytic']:.3e}"
    )
    print(
        f"GD, AD subgradient:       rel. error to Scikit-Learn = {check['error_ad']:.3e}"
    )
    print(
        f"GD analytic vs GD AD:     rel. difference            = {check['diff_analytic_ad']:.3e}"
    )

    # 2. lambda sweep
    lambda_vals, mse_vals, r2_vals, params_own, params_sk = run_lasso_over_lambda(cfg)
    n_train = len(get_lasso_data(cfg)[2])

    print(f"\nLasso over lambda (degree {degree}), own GD vs Scikit-Learn")
    print(
        "-----------------------------------------------------------------------------"
    )
    print(
        f"{'lambda':>10} {'sklearn alpha':>14} {'MSE own':>10} {'MSE sklearn':>12} "
        f"{'nonzero sklearn':>16} {'max|dtheta|':>12}"
    )
    for i, lam in enumerate(lambda_vals):
        n_nonzero = int(np.sum(params_sk[i][1:] != 0))
        max_diff = np.max(np.abs(params_own[i] - params_sk[i]))
        print(
            f"{lam:10.2e} {lam / (2 * n_train):14.2e} {mse_vals[0, i]:10.5f} "
            f"{mse_vals[1, i]:12.5f} {n_nonzero:16d} {max_diff:12.2e}"
        )

    plot.plot_mse_and_r2(
        lambda_vals,
        mse_vals,
        r2_vals,
        variable="lambda",
        method="lasso",
        labels=["Own GD", "Scikit-Learn"],
        suffix=f"_degree_{degree}",
    )
    plot.plot_parameters(lambda_vals, params_own, variable="lambda", method="lasso")
    plot.plot_parameters(
        lambda_vals, params_sk, variable="lambda", method="lasso_sklearn"
    )

    # 3. optimizers

    learning_rates, iterations, final_errors, cost_gaps = run_lasso_optimizers(cfg)
    best = best_learning_rates(learning_rates, final_errors)

    print(
        f"\nLasso optimizers (degree {degree}, lambda = {lmbda}, target rel. error {target:.0e}, "
        f"max {max_iter} iterations), reference: Scikit-Learn"
    )
    print(
        "-----------------------------------------------------------------------------"
    )
    print(
        f"{'Optimizer':<10} {'best lr':>9} {'final error':>12} {'cost gap':>11} "
        f"{'iter. to target':>16}   {'lr range reaching target':<24}"
    )
    for i, name in enumerate(OPTIMIZER_NAMES):
        if np.isnan(best[i]):  # diverged for every learning rate
            print(f"{name:<10} {'-':>9} {'diverged':>12}")
            continue
        j = list(learning_rates).index(best[i])  # column of the best learning rate
        reached = learning_rates[np.isfinite(iterations[i])]
        if len(reached) > 0:
            lr_range = ", ".join(f"{lr:.0e}" for lr in reached)
        else:
            lr_range = "never"
        n_iter = "-" if np.isnan(iterations[i, j]) else f"{int(iterations[i, j])}"
        print(
            f"{name:<10} {best[i]:9.1e} {final_errors[i, j]:12.2e} {cost_gaps[i, j]:11.2e} "
            f"{n_iter:>16}   {lr_range:<24}"
        )

    # GD stability limit for the plots (OLS Hessian; the l1 term has no curvature)
    X_train, _, y_train, _ = get_lasso_data(cfg)
    X_scaled, _ = sklearn_lasso_fit(X_train, y_train, lmbda).scale(X_train, y_train)
    gamma_max = max_learning_rate(X_scaled, method="lasso")[1]

    plot.plot_learning_rate_sensitivity(
        learning_rates,
        iterations,
        OPTIMIZER_NAMES,
        ylabel=f"Iterations to reach relative error {target:.0e}",
        method="lasso",
        name="iterations_to_target",
        gamma_max=gamma_max,
    )
    plot.plot_learning_rate_sensitivity(
        learning_rates,
        final_errors,
        OPTIMIZER_NAMES,
        ylabel=f"Relative error after {max_iter} iterations",
        method="lasso",
        name="final_error",
        gamma_max=gamma_max,
    )

    converged = [
        (name, lr) for name, lr in zip(OPTIMIZER_NAMES, best) if not np.isnan(lr)
    ]  # skip optimizers that diverged for every learning rate
    histories = [lasso_error_vs_sklearn(cfg, name, lr)[0] for name, lr in converged]
    labels = [rf"{name} ($\gamma$ = {lr})" for name, lr in converged]
    plot.plot_convergence(
        histories,
        labels,
        method="lasso",
        target=target,
        ylabel="Relative error to Scikit-Learn",
        ncol=2,
        smooth_window=100,
    )

    # 4.
    results = run_method_comparison(cfg)
    print(f"\nOLS, Ridge and Lasso (degree {degree}, lambda = {lmbda})")
    print(
        "-----------------------------------------------------------------------------"
    )
    print(
        f"{'Model':<22} {'test MSE':>10} {'|theta_j| < 1e-3':>18}   coefficients (theta_1, ...)"
    )
    for label, test_mse, theta in results:
        n_small = int(np.sum(np.abs(theta[1:]) < 1e-3))
        coefs = " ".join(f"{c:7.3f}" for c in theta[1:])
        print(f"{label:<22} {test_mse:10.5f} {n_small:18d}   {coefs}")

    plot.plot_coefficient_comparison(results, lmbda=lmbda, degree=degree)
