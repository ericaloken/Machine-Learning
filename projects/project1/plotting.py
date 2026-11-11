# plotting.py


import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy
from pathlib import Path
import numpy as np
from matplotlib.ticker import LogFormatterSciNotation
from scipy.ndimage import uniform_filter1d  # to smooth the curves for noisy plots

FIG_DIR = Path(__file__).resolve().parent / "figures"  # figures/ next to this file
PNG_DIR = FIG_DIR / "pngs"  # PNGs for the notebook
PNG_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use(Path(__file__).resolve().parents[2] / "style.mplstyle")


def save_figure(fig, filename, Tight=True):
    for ax in fig.axes:
        if (
            ax.get_yscale() == "log"
        ):  # label minor ticks when the axis spans only a decade or two
            ax.yaxis.set_minor_formatter(
                LogFormatterSciNotation(minor_thresholds=(2, 0.4))
            )
    if Tight:
        fig.tight_layout()
    fig.savefig(FIG_DIR / filename, bbox_inches="tight")  # include the legend
    fig.savefig(
        PNG_DIR / filename.replace(".pdf", ".png"), dpi=150, bbox_inches="tight"
    )
    print(f"Saved: figures/{filename}")
    plt.close(fig)


def legend_below(ax, ncol=3, **kwargs):
    """
    Place the legend centered below the plot, under the x-axis label, and make the
    figure taller so the plot itself keeps its size.

    input:  ax (Axes)       - the axes to add the legend to
            ncol (int)      - number of legend columns
            **kwargs        - passed on to ax.legend, e.g. handles
    """
    handles = kwargs.pop("handles", None)
    if handles is None:
        handles, _ = ax.get_legend_handles_labels()
    if len(handles) == 0:
        return

    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    below_axes = offset_copy(ax.transAxes, fig=fig, y=-40, units="points")

    ncol = min(ncol, len(handles))
    while True:
        legend = ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            bbox_transform=below_axes,
            ncol=ncol,
            frameon=False,
            **kwargs,
        )
        too_wide = (
            legend.get_window_extent(renderer).width
            > ax.get_window_extent(renderer).width
        )
        if not too_wide or ncol == 1:
            break
        legend.remove()
        ncol -= 1

    n_rows = int(np.ceil(len(handles) / ncol))
    fig.set_figheight(fig.get_figheight() + 0.3 + 0.25 * n_rows)


def format_x_axis(ax, variable) -> None:
    """
    Format the x-axis for plotting.
    """
    if variable in ("degree", "degrees"):
        ax.set_xlabel("Polynomial degree")

    elif variable in ("lambda", "lambdas"):
        ax.set_xlabel(r"$\lambda$")
        ax.set_xscale("log")

    else:
        raise ValueError("variable must be degree(s) or lambda(s)")


def plot_mse_and_r2(
    x_values,
    mse_values,
    r2_values,
    variable="degree",
    method="ols",
    labels=None,
    suffix="",
) -> None:
    """
    Plot MSE (left y-axis, log scale) and R^2 (right y-axis) against polynomial
    degree or lambda in one figure, and save it.

    One curve: MSE and R^2 get their own colour, and each y-axis is coloured to match.
    Several curves (e.g. different n, sigma or lambda): colour = curve,
    solid line with filled circles = MSE, dashed line with open squares = R^2.

    input:  x_values (array-like)        - polynomial degrees or lambda values
            mse_values (array-like)      - shape (n_points,) or (n_curves, n_points)
            r2_values (array-like)       - same shape as mse_values
            variable (str)               - "degree" or "lambda"
            method (str)                 - regression method, used in the filename
            labels (list or None)        - name of each curve (needed for several curves)
            suffix (str)                 - extra text appended to the filename
    """
    mse_values = np.atleast_2d(mse_values)  # one curve -> shape (1, n_points)
    r2_values = np.atleast_2d(r2_values)
    n_curves = mse_values.shape[0]

    if mse_values.shape != r2_values.shape or mse_values.shape[1] != len(x_values):
        raise ValueError(
            "mse_values and r2_values must both have shape (n_curves, len(x_values))"
        )
    if labels is not None and len(labels) != n_curves:
        raise ValueError(f"got {len(labels)} labels for {n_curves} curves")

    single = n_curves == 1
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax_mse = plt.subplots()
    ax_r2 = ax_mse.twinx()  # second y-axis on the right, same x-axis

    for i in range(n_curves):
        mse_color = colors[0] if single else colors[i % len(colors)]
        r2_color = colors[1] if single else colors[i % len(colors)]
        ax_mse.plot(
            x_values,
            mse_values[i],
            alpha=0.9,
            color=mse_color,
            linestyle="-",
            marker="o",
        )
        ax_r2.plot(
            x_values,
            r2_values[i],
            color=r2_color,
            linestyle="--",
            alpha=0.7,
            marker="o",
            markerfacecolor="none",
        )

    format_x_axis(ax_mse, variable)
    ax_mse.set_ylabel("MSE")
    ax_r2.set_ylabel(r"$R^2$")
    ax_mse.grid()

    # legend: one entry per curve (several curves), plus what solid and dashed mean
    handles = []
    if not single and labels is not None:
        handles += [
            plt.Line2D(
                [], [], alpha=0.9, color=colors[i % len(colors)], label=labels[i]
            )
            for i in range(n_curves)
        ]
    handles += [
        plt.Line2D(
            [],
            [],
            alpha=0.7,
            color=colors[0] if single else "black",
            linestyle="-",
            marker="o",
            label="MSE (left axis)",
        ),
        plt.Line2D(
            [],
            [],
            alpha=0.5,
            color=colors[1] if single else "black",
            linestyle="--",
            marker="o",
            markerfacecolor="none",
            label=r"$R^2$ (right axis)",
        ),
    ]
    legend_below(ax_mse, handles=handles)

    filename = f"{method}_mse_r2_vs_{variable}{suffix}.pdf"
    save_figure(fig, filename)


def plot_fit_panels(grid, f_grid, panels, filename, ncols=3, ylim=(-0.3, 1.3)) -> None:
    """
    Fitted curves against the data, one panel for each setting (here data set size and
    degree), with training points, test points and the true function. One shared legend
    below. Curves that leave the y-range are clipped.

    input:  grid (array)             - x values where the curves are evaluated
            f_grid (array)           - true function on the grid
            panels (list of dict)    - one dict per panel with the keys "title" (str),
                                       "x_train", "y_train", "x_test", "y_test" (arrays)
                                       and "curves" (list of (label, color, y on grid))
            filename (str)           - output name, ending in ".pdf"
            ncols (int)              - number of panel columns
            ylim (tuple)             - y-range of all panels
    """
    nrows = int(np.ceil(len(panels) / ncols))
    width, height = plt.rcParams["figure.figsize"]
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2 * width, 0.85 * height * nrows),
        sharex=True,
        sharey=True,
        layout="constrained",
        squeeze=False,
    )

    for ax, panel in zip(axes.flat, panels):
        ax.scatter(
            panel["x_train"],
            panel["y_train"],
            s=4,
            color="0.55",
            label="Training data",
            zorder=1,
        )
        ax.scatter(
            panel["x_test"],
            panel["y_test"],
            s=8,
            facecolor="none",
            edgecolor="C3",
            linewidth=0.5,
            label="Test data",
            zorder=1,
        )
        ax.plot(
            grid,
            f_grid,
            color="black",
            linestyle="--",
            linewidth=0.9,
            label="True $f(x)$",
            zorder=3,
        )
        for label, color, curve in panel["curves"]:
            ax.plot(grid, curve, color=color, label=label, zorder=2)
        ax.set_ylim(*ylim)
        ax.set_title(panel["title"])
        ax.grid()

    for ax in axes.flat[len(panels) :]:  # hide unused panels
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(r"$x$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$y$")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=len(handles),
        frameon=False,
        markerscale=1.5,
    )

    save_figure(fig, filename, Tight=False)


def plot_shrinkage_factors(singular_values, lambdas, degree) -> None:
    """
    Ridge shrinkage factor sigma^2 / (sigma^2 + lambda) against the singular value
    (as Fig. 3.1 in the lecture notes), with the singular values of our own scaled
    design matrix marked on each curve.

    input:  singular_values (array)  - singular values of the scaled training design matrix
            lambdas (list)           - Ridge regularization parameters
            degree (int)             - polynomial degree, used in the filename
    """
    fig, ax = plt.subplots()

    # smooth curves over a range that covers all singular values
    sigma = np.logspace(
        np.log10(singular_values.min()) - 1, np.log10(singular_values.max()) + 1, 300
    )
    for lmbda in lambdas:
        (line,) = ax.plot(
            sigma, sigma**2 / (sigma**2 + lmbda), label=rf"$\lambda = {lmbda:.0e}$"
        )
        # our design matrix: one marker per singular-value mode
        factors = singular_values**2 / (singular_values**2 + lmbda)
        ax.plot(singular_values, factors, "o", color=line.get_color(), markersize=4)

    ax.set_xscale("log")
    ax.set_xlabel(r"Singular value $\sigma_i$")
    ax.set_ylabel(r"Shrinkage factor $\sigma_i^2 / (\sigma_i^2 + \lambda)$")
    legend_below(ax)
    ax.grid()

    filename = f"ridge_shrinkage_factors_degree_{degree}.pdf"
    save_figure(fig, filename)


def plot_parameters(
    x_values, parameter_values, variable="degree", method="ols", split_at=None
) -> None:
    """
    Plot and save fitted coefficients against polynomial degree or lambda.

    With split_at, the figure gets two panels side by side: x <= split_at on the
    left and x >= split_at on the right, each with its own y-range, so small
    coefficients are visible next to very large ones. One legend below both.

    input:  x_values (array-like)           - degrees or lambdas
            parameter_values (list)         - fitted coefficients (theta0, theta1, ...) for each x
            variable (str)                  - "degree" or "lambda"
            method (str)                    - regression method, used in the filename
            split_at (float, optional)      - x-value where the figure is split into two panels
    """
    x_values = np.asarray(x_values)  # a range can't be compared with split_at
    n = max(len(parameters) for parameters in parameter_values)
    coefficients = np.full(
        (len(x_values), n), np.nan
    )  # rows = models, columns = coefficients
    for i, parameters in enumerate(parameter_values):
        coefficients[i, : len(parameters)] = parameters

    if split_at is None:
        fig, ax = plt.subplots()
        axes, masks = [ax], [np.ones(len(x_values), dtype=bool)]
    else:
        width, height = plt.rcParams["figure.figsize"]
        fig, axes = plt.subplots(
            1, 2, figsize=(2 * width, height), layout="constrained"
        )
        # both panels include split_at, so the curves continue across the split
        masks = [x_values <= split_at, x_values >= split_at]

    colors = plt.get_cmap("tab20")

    # plot each coefficient across models, in each panel only its part of the x-range
    for ax, mask in zip(axes, masks):
        for i in range(n):
            ax.plot(
                x_values[mask],
                coefficients[mask, i],
                marker="o",
                markersize=4,
                color=colors(i % 20),
                label=rf"$\theta_{{{i}}}$",
            )
        format_x_axis(ax, variable)
        ax.grid()

    axes[0].set_ylabel("Standardized coefficient value")

    if split_at is None:
        legend_below(axes[0], ncol=5)
    else:
        # title on each panel with its range
        if variable in ("lambda", "lambdas"):
            name, value = r"\lambda", rf"10^{{{np.log10(split_at):.0f}}}"
        else:
            name, value = "d", f"{split_at:g}"
        axes[0].set_title(rf"${name} \leq {value}$")
        axes[1].set_title(rf"${name} \geq {value}$")

        # one legend below both panels
        handles, _ = axes[0].get_legend_handles_labels()
        fig.legend(
            handles=handles,
            loc="outside lower center",
            ncol=int(np.ceil(n / 2)),
            frameon=False,
        )

    filename = f"{method}_parameters_vs_{variable}.pdf"
    save_figure(fig, filename, Tight=False)


def plot_train_test_mse(degrees, train_mse, test_mse, sigma=None) -> None:
    """
    Plot training and test MSE (recreate Figure 2.11). Finds the best polynomial degree by locating
    which degree gives lovest test MSE.
    """
    idx = np.argmin(test_mse)
    best_deg = degrees[idx]
    # best_mse = test_mse[idx]

    fig, ax = plt.subplots()
    ax.plot(degrees, train_mse, marker="o", label="Training")
    ax.plot(degrees, test_mse, marker="s", label="Test")

    plt.axvline(
        best_deg, color="gray", alpha=0.8, linestyle=":", label=f"degree {best_deg}"
    )
    if sigma is not None:
        ax.axhline(
            sigma**2,
            color="gray",
            alpha=0.8,
            linestyle="--",
            label=rf"$\sigma^2 = {sigma**2:g}$",
        )
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    legend_below(ax)
    # ax.grid()
    filename = f"train_test_mse_vs_degree.pdf"
    save_figure(fig, filename)


def plot_bias_variance_both(
    degrees, mse, variance, bias_f, bias_y, sigma, method="ols", suffix=""
) -> None:
    """
    Plot and save test MSE, variance and squared bias against polynomial degree,
    with the bias measured against both the true function f and the noisy data y.
    MSE and variance do not depend on the bias reference, so they are drawn once.
    The noise floor sigma^2 is drawn as a horizontal line: with y as reference,
    bias^2 flattens out at sigma^2.

    input:  degrees (array-like)  - polynomial degrees
            mse (array-like)      - test mean squared error for each degree
            variance (array-like) - variance for each degree
            bias_f (array-like)   - squared bias relative to the true function f
            bias_y (array-like)   - squared bias relative to the noisy data y
            sigma (float)         - noise standard deviation, the line is drawn at sigma^2
            method (str)          - regression method, either "ols" or "ridge"
            suffix (str)          - additions to filename
    """
    if method not in ["ols", "ridge"]:
        raise ValueError("method must be either 'ols' or 'ridge'")

    fig, ax = plt.subplots()
    ax.plot(degrees, mse, marker="o", label="Test error")
    ax.plot(degrees, variance, marker="^", label="Variance")
    ax.plot(degrees, bias_f, marker="s", label=r"True Bias$^2$ ($f$)")
    ax.plot(
        degrees,
        bias_y,
        marker="s",
        linestyle="--",
        markerfacecolor="none",
        label=r"Measured Bias$^2$ ($y$)",
    )
    ax.axhline(
        sigma**2,
        color="gray",
        alpha=0.8,
        linestyle=":",
        label=rf"$\sigma^2 = {sigma**2:g}$",
    )
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("MSE decomposition")
    ax.set_yscale("log")
    legend_below(ax)
    ax.grid()

    filename = f"{method}_bias_variance_vs_degree_both{suffix}.pdf"
    save_figure(fig, filename)


def plot_noise_floor(
    degrees, mse, bias_f, variance, bias_y, sigma, method="ols", suffix=""
) -> None:
    """
    Plot estimates of the irreducible noise sigma^2 against polynomial degree:
    MSE - Bias^2_f - Var and Bias^2_y - Bias^2_f, with sigma^2 as a horizontal line.
    Both are equal to sigma^2 only in expectation, so they fluctuate around it, but
    should be flat as a function of model complexity. Linear y-axis, since the
    differences can be negative.

    input:  degrees (array-like)  - polynomial degrees
            mse (array-like)      - test MSE (measured against y)
            bias_f (array-like)   - squared bias with the true function f as reference
            variance (array-like) - variance of the predictions
            bias_y (array-like)   - squared bias with the noisy data y as reference
            sigma (float)         - standard deviation of the noise
            method (str)          - regression method, used in the filename
            suffix (str)          - extra text appended to the filename
    """
    degrees = np.asarray(degrees)
    mse, bias_f = np.asarray(mse), np.asarray(bias_f)
    variance, bias_y = np.asarray(variance), np.asarray(bias_y)

    fig, ax = plt.subplots()
    ax.plot(
        degrees,
        mse - bias_f - variance,
        marker="o",
        label=r"MSE $-$ Bias$^2_f$ $-$ Var",
    )
    ax.plot(
        degrees,
        bias_y - bias_f,
        marker="s",
        linestyle="--",
        label=r"Bias$^2_y$ $-$ Bias$^2_f$",
    )
    ax.axhline(
        sigma**2,
        color="gray",
        alpha=0.8,
        linestyle=":",
        label=rf"$\sigma^2 = {sigma**2:g}$",
    )

    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel(r"Estimate of $\sigma^2$")
    legend_below(ax)
    ax.grid()

    filename = f"{method}_noise_floor_vs_degree{suffix}.pdf"
    save_figure(fig, filename)


def plot_bias_variance_multiple(
    degrees,
    bias_values,
    variance_values,
    mse_values,
    labels,
    varying_parameter="n",
    method="ols",
    suffix="",
    bias_reference="y",
    ymax=None,
) -> None:
    """
    One wide figure with three panels, test MSE (a), squared bias (b) and variance (c)
    against polynomial degree, for several values of a parameter (e.g. n), with one
    shared legend below.

    bias_values, variance_values and mse_values have shape (n_curves, n_degrees).
    ymax (float, optional) cuts the y-axes at this value, so curves that blow up do not
    hide the region of interest.
    """
    bias_values = np.asarray(bias_values)
    variance_values = np.asarray(variance_values)
    mse_values = np.asarray(mse_values)

    if not (bias_values.shape == variance_values.shape == mse_values.shape):
        raise ValueError("bias, variance and MSE arrays must have the same shape")
    if len(labels) != bias_values.shape[0]:
        raise ValueError("labels must have one label per curve")

    bias_label = r"Measured Bias$^2$" if bias_reference == "y" else r"True Bias$^2$"
    panels = [
        ("MSE", mse_values, "(a)"),
        (bias_label, bias_values, "(b)"),
        ("Variance", variance_values, "(c)"),
    ]

    width, height = plt.rcParams["figure.figsize"]
    fig, axes = plt.subplots(1, 3, figsize=(2 * width, height), layout="constrained")

    for ax, (ylabel, values, letter) in zip(axes, panels):
        for i, label in enumerate(labels):
            ax.plot(degrees, values[i], marker="o", markersize=3, label=label)
        ax.set_xlabel("Polynomial degree")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        if ymax is not None:
            ax.set_ylim(top=ymax)
        ax.set_title(letter, loc="left")
        ax.grid()

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="outside lower center",
        ncol=len(labels),
        frameon=False,
    )

    filename = (
        f"{method}_bias_variance_vs_degree_different_{varying_parameter}{suffix}.pdf"
    )
    save_figure(fig, filename, Tight=False)


def plot_cv_bootstrap_mse_comparison(
    degrees,
    mse_bootstrap,
    cv_results,
    method="ols",
) -> None:
    """
    Plot bootstrap MSE and CV MSE for multiple numbers of folds
    against polynomial degree in the same figure.
    """

    degrees = np.asarray(degrees)

    best_boot = degrees[np.argmin(mse_bootstrap)]

    fig, ax = plt.subplots()

    # Bootstrap
    ax.plot(
        degrees,
        mse_bootstrap,
        marker="o",
        label=rf"Bootstrap ($d=${best_boot})",
    )

    # Cross-validation
    for k, (mse_cv, std_cv) in cv_results.items():
        best_cv = degrees[np.argmin(mse_cv)]

        ax.plot(
            degrees,
            mse_cv,
            marker="s",
            label=rf"{k}-fold ($d=${best_cv})",
        )

    format_x_axis(ax, "degree")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    legend_below(ax)
    ax.grid()

    filename = f"{method}_cv_bootstrap_vs_degree.pdf"
    save_figure(fig, filename)


def plot_cv_mse_multiple(
    x_values,
    results,
    variable,
    method,
    suffix="",
) -> None:
    """
    Plot CV MSE curves for several values of k.

    Parameters
    ----------
    x_values : array-like
        Polynomial degrees or lambda values.

    results : dict[int, tuple[array-like, array-like]]
        Mapping from k to (mean_mse, fold_mse_std).

    variable : {"degree", "lambda"}
        Quantity on the x-axis.

    method : {"ols", "ridge"}
        Regression method.
    """
    fig, ax = plt.subplots()

    for k, (mse, std) in results.items():
        ax.plot(
            x_values,
            mse,
            marker="o",
            label=rf"${k}$-fold CV",
        )

        ax.fill_between(
            x_values,
            mse - std,
            mse + std,
            alpha=0.25,
        )

    if variable == "degree":
        ax.set_xlabel("Polynomial degree")
    elif variable == "lambda":
        ax.set_xlabel(r"Regularization parameter $\lambda$")
        ax.set_xscale("log")
    else:
        raise ValueError("variable must be either 'degree' or 'lambda'")

    ax.set_ylabel("Cross-validation MSE")
    ax.set_yscale("log")
    ax.grid()
    legend_below(ax)

    filename = f"{method}_cv_mse_{variable}_k_comparison{suffix}.pdf"
    save_figure(fig, filename)


def plot_convergence(
    histories,
    labels,
    method="ols",
    target=None,
    ylabel="Relative error to closed form",
    smooth_window=1,
    ncol=2,
) -> None:
    """
    Plot the relative error to the closed-form solution against iteration
    for several optimizers in the same figure.

    input:  histories (list of arrays)  - relative error in each iteration, one array per optimizer
            labels (list of str)        - name of each curve
            method (str)                - "ols" or "ridge", used in the filename
            target (float or None)      - target accuracy, drawn as a horizontal line
            ylabel (str)                - label for the y-axis
            smooth_window (int)         - window size for smoothing the curves (1 = no smoothing)
            ncol (int)                   - number of columns in the legend
    """
    fig, ax = plt.subplots()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (errors, label) in enumerate(zip(histories, labels)):
        errors = np.asarray(errors, dtype=float)
        iterations = np.arange(1, len(errors) + 1)
        color = colors[i % len(colors)]

        if (
            smooth_window > 1
        ):  # one curve for smooth and one for raw noise with low alpha
            ax.plot(
                iterations, errors, color=color, alpha=0.15, linewidth=0.5
            )  # raw noise
            smoothed = 10 ** uniform_filter1d(
                np.log10(errors), size=smooth_window, mode="nearest"
            )
            ax.plot(iterations, smoothed, color=color, linewidth=1.6, label=label)
        else:
            ax.plot(iterations, errors, color=color, label=label)

    if target is not None:
        ax.axhline(target, color="gray", linestyle="--", label=f"Target {target:.0e}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    # ax.set_xscale("log")
    legend_below(ax, ncol=ncol)
    ax.grid()

    save_figure(fig, f"{method}_convergence_optimizers.pdf")


def plot_learning_rate_sensitivity(
    learning_rates, values, labels, ylabel, method="ols", name="", gamma_max=None
) -> None:
    """
    Plot a quantity (e.g. iterations to reach a target accuracy, or the final error)
    against the learning rate, one curve per optimizer. NaN values (target not
    reached or diverged) are left out of the curve.

    input:  learning_rates (array)      - learning rates
            values (2D array)           - shape (n_optimizers, n_learning_rates)
            labels (list of str)        - optimizer names
            ylabel (str)                - label for the y-axis
            method (str)                - "ols" or "ridge", used in the filename
            name (str)                  - what is plotted, used in the filename
            gamma_max (float or None)   - GD stability limit 2/lambda_max, drawn as a vertical line
    """
    fig, ax = plt.subplots()

    for i, label in enumerate(labels):
        ax.plot(learning_rates, values[i], marker="o", label=label)

    if gamma_max is not None:
        ax.axvline(gamma_max, color="gray", linestyle=":", label=r"$\gamma_{max}$ (GD)")

    ax.set_xlabel(rf"Initial learning rate $\gamma$")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.set_yscale("log")
    legend_below(ax)
    ax.grid()

    filename = f"{method}_{name}_vs_learning_rate.pdf"
    save_figure(fig, filename)


def plot_learning_rate_sensitivity_ols_ridge(
    learning_rates, values_by_method, labels, ylabel, name="", gamma_max=None
) -> None:
    """
    Same as plot_learning_rate_sensitivity, but OLS and Ridge in one plot:
    one color per optimizer, solid lines for OLS and dashed lines for Ridge.

    input:  learning_rates (array)      - learning rates
            values_by_method (dict)     - {"ols": 2D array, "ridge": 2D array},
                                        each of shape (n_optimizers, n_learning_rates)
            labels (list of str)        - optimizer names
            ylabel (str)                - label for the y-axis
            name (str)                  - what is plotted, used in the filename
            gamma_max (float or None)   - GD stability limit 2/lambda_max, drawn as a vertical line
    """
    fig, ax = plt.subplots()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    styles = {"ols": ("-", "o"), "ridge": ("--", "s")}  # (line style, marker)

    for method, values in values_by_method.items():
        linestyle, marker = styles[method]
        for i in range(len(labels)):
            ax.plot(
                learning_rates,
                values[i],
                linestyle=linestyle,
                marker=marker,
                color=colors[i],
                markerfacecolor="none" if method == "ridge" else colors[i],
            )

    # legend: color = optimizer, line style = method
    handles = [
        plt.Line2D([], [], color=colors[i], label=label)
        for i, label in enumerate(labels)
    ]
    handles += [
        plt.Line2D(
            [],
            [],
            color="black",
            linestyle=linestyle,
            marker=marker,
            markerfacecolor="none" if method == "ridge" else "black",
            label=method.upper(),
        )
        for method, (linestyle, marker) in styles.items()
    ]
    if gamma_max is not None:
        handles.append(
            ax.axvline(
                gamma_max, color="gray", linestyle=":", label=r"$\gamma_{max}$ (GD)"
            )
        )

    ax.set_xlabel(r"Initial learning rate $\gamma$")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.set_yscale("log")
    legend_below(ax, handles=handles)
    ax.grid()

    filename = f"ols_ridge_{name}_vs_learning_rate.pdf"
    save_figure(fig, filename)


def plot_coefficient_comparison(results, lmbda, degree) -> None:
    """
    Bar chart of the standardized coefficients of several fitted models, e.g. OLS,
    Ridge and Lasso, to show shrinkage (Ridge) and exact zeros (Lasso).

    input:  results (list of tuples)    - (label, test MSE, theta) for each model
            lmbda (float)               - regularization parameter, shown in the title
            degree (int)                - polynomial degree, shown in the title
    """
    fig, ax = plt.subplots()

    n_models = len(results)
    n_coef = len(results[0][2])
    width = 0.8 / n_models  # all bars for one coefficient share 80% of the space
    positions = np.arange(n_coef)  # one group of bars per coefficient

    for k, (label, _, theta) in enumerate(results):
        offset = (
            k - (n_models - 1) / 2
        ) * width  # centre the group around each position
        ax.bar(positions + offset, theta, width, label=label)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels([rf"$\theta_{{{i}}}$" for i in range(n_coef)])
    ax.set_ylabel("Standardized coefficient value")
    legend_below(ax)
    ax.grid(axis="y")

    filename = "ols_ridge_lasso_coefficients.pdf"
    save_figure(fig, filename)


def plot_error_vs_epoch(
    histories_solid,
    histories_dashed,
    labels,
    solid_label,
    dashed_label,
    method="ols",
    name="",
    target=None,
    smooth_window=1,
) -> None:
    """
    Plot the relative error to the closed-form solution against epoch for several
    optimizers, in two variants (solid and dashed lines, same colour per optimizer),
    e.g. SGD vs full batch, or constant vs decaying learning rate.

    input:  histories_solid (list of arrays)    - error per epoch, one array per optimizer
            histories_dashed (list of arrays)   - same, for the second variant
            labels (list of str)                - optimizer names
            solid_label, dashed_label (str)     - what the two line styles mean
            method (str)                        - "ols" or "ridge", used in the filename
            name (str)                          - what is compared, used in the filename
            target (float or None)              - target accuracy, drawn as a horizontal line
    """
    fig, ax = plt.subplots()
    colors = plt.get_cmap("tab10")

    for i, label in enumerate(labels):
        for history, linestyle, lab in (
            (histories_solid[i], "-", label),
            (histories_dashed[i], "--", None),
        ):
            history = np.asarray(history, dtype=float)
            epochs = np.arange(1, len(history) + 1)

            if smooth_window > 1:
                ax.plot(
                    epochs,
                    history,
                    color=colors(i),
                    linestyle=linestyle,
                    alpha=0.15,
                    linewidth=0.5,
                )  # raw noise
                smoothed = 10 ** uniform_filter1d(
                    np.log10(history), size=smooth_window, mode="nearest"
                )
                ax.plot(
                    epochs, smoothed, color=colors(i), linestyle=linestyle, label=lab
                )
            else:
                ax.plot(
                    epochs, history, color=colors(i), linestyle=linestyle, label=lab
                )

    # legend entries for the two line styles
    ax.plot([], [], color="black", label=solid_label)
    ax.plot([], [], color="black", linestyle="--", label=dashed_label)

    if target is not None:
        ax.axhline(target, color="gray", linestyle=":", label=f"Target {target:.0e}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative error to closed form")
    # ax.set_xscale("log")
    ax.set_yscale("log")
    legend_below(ax)
    ax.grid()

    filename = f"{method}_sgd_error_vs_epoch_{name}.pdf"
    save_figure(fig, filename)


def plot_error_vs_batch_size(
    batch_sizes, final_errors, labels, n_epochs, method="ols"
) -> None:
    """
    Plot the relative error after a fixed number of epochs against the mini-batch size,
    one curve per optimizer. NaN values (diverged runs) are left out.

    input:  batch_sizes (array)         - mini-batch sizes (the full data set size for full batch)
            final_errors (2D array)     - shape (n_optimizers, n_batch_sizes)
            labels (list of str)        - optimizer names
            n_epochs (int)              - number of epochs, shown in the axis label
            method (str)                - "ols" or "ridge", used in the filename
    """
    fig, ax = plt.subplots()

    for i, label in enumerate(labels):
        ax.plot(batch_sizes, final_errors[i], marker="o", label=label)

    ax.set_xlabel("Mini-batch size $M$")
    ax.set_ylabel(f"Relative error after {n_epochs} epochs")
    # ax.set_xscale("log")
    ax.set_yscale("log")
    legend_below(ax)
    ax.grid()

    filename = f"{method}_sgd_error_vs_batch_size.pdf"
    save_figure(fig, filename)


def plot_cv_mse_methods(curves, labels, k, styles=None) -> None:
    """
    Plot the cross-validated MSE against polynomial degree for several methods
    in one figure (for Ridge and Lasso: the best lambda at each degree).

    input:  curves (list of tuples)     - (degrees, mse) for each method
            labels (list of str)        - name of each curve
            k (int)                     - number of folds, used in the label and filename
            styles (list of str)        - optional line styles, e.g. "--" for a reference curve
    """
    fig, ax = plt.subplots()

    for i, ((degrees, mse), label) in enumerate(zip(curves, labels)):
        linestyle = "-" if styles is None else styles[i]
        ax.plot(degrees, mse, marker="o", linestyle=linestyle, label=label, alpha=0.8)

    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel(f"{k}-fold cross-validated MSE")
    ax.set_yscale("log")
    legend_below(ax)
    ax.grid()

    filename = f"model_selection_cv_mse_vs_degree_k{k}.pdf"
    save_figure(fig, filename)


def plot_cv_heatmap(degrees, lambdas, mse_grid, method, k) -> None:
    """
    Heatmap of the cross-validated MSE over polynomial degree and lambda,
    with the best combination marked.

    input:  degrees (array)         - polynomial degrees (rows of mse_grid)
            lambdas (array)         - lambda values (columns of mse_grid)
            mse_grid (2D array)     - CV MSE, shape (n_degrees, n_lambdas)
            method (str)            - "ridge" or "lasso", used in the filename
            k (int)                 - number of folds
    """
    fig, ax = plt.subplots()

    image = ax.imshow(np.log10(mse_grid), origin="lower", aspect="auto", cmap="viridis")
    fig.colorbar(image, ax=ax, label=rf"$\log_{{10}}$ ({k}-fold CV MSE)")

    best_row, best_col = np.unravel_index(np.nanargmin(mse_grid), mse_grid.shape)
    exponents = np.log10(lambdas)
    ax.scatter(
        best_col,
        best_row,
        marker="*",
        color="red",
        s=80,
        zorder=3,
        label=rf"Best: $d={degrees[best_row]}$, $\lambda=10^{{{exponents[best_col]:g}}}$",
    )

    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels(  # every second label, so they do not overlap
        [f"$10^{{{e:g}}}$" if i % 2 == 0 else "" for i, e in enumerate(exponents)]
    )
    ax.set_yticks(np.arange(len(degrees)))
    ax.set_yticklabels(
        [str(d) if d % 5 == 0 or d == degrees[0] else "" for d in degrees]
    )
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Polynomial degree")
    legend_below(ax, scatterpoints=1, markerscale=0.8)

    filename = f"{method}_cv_heatmap_k{k}.pdf"
    save_figure(fig, filename)
