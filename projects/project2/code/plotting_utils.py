# plotting.py


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.transforms import offset_copy
from pathlib import Path
import numpy as np
from matplotlib.ticker import LogFormatterSciNotation
from scipy.ndimage import uniform_filter1d  # to smooth the curves for noisy plots

FIG_DIR = Path(__file__).resolve().parent / "figures"  # figures/ next to this file
plt.style.use(Path(__file__).resolve().parents[3] / "style.mplstyle")

# heatmap colors: blue for non-zero values, starting at a visible light blue so the
# smallest values do not look white, and a separate gray for cells that are exactly 0
HEATMAP_CMAP = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.15, 1.0, 256)))
ZERO_COLOR = "lightgray"


def mark_zeros(ax, values):
    """Paint the cells that are exactly 0 in ZERO_COLOR, on top of a seaborn heatmap of values."""
    values = np.asarray(values, dtype=float)
    zero = values == 0
    if zero.any():
        # seaborn draws cell (i, j) at x = j..j+1, y = i..i+1, so pcolormesh lines up with it
        ax.pcolormesh(np.ma.masked_where(~zero, values), cmap=ListedColormap([ZERO_COLOR]))
        # seaborn chose the annotation color for the cell below, use dark text on the gray
        for text in ax.texts:
            x, y = text.get_position()
            if zero[int(y), int(x)]:
                text.set_color("black")


def highlight_cell(ax, x, y, color="red", linestyle="-", linewidth=1.5):
    """
    Box around the heatmap cell with lower left corner (x, y), i.e. column x and row y.
    The line is clipped to the cell, so it lies fully inside it and has the same
    width on all four sides, also for cells at the edge of the plot.
    """
    box = Rectangle((x, y), 1, 1, fill=False, edgecolor=color, linestyle=linestyle,
                    linewidth=2 * linewidth, zorder=10)  # half of the line is clipped away
    ax.add_patch(box)
    box.set_clip_path(box)

def save_figure(fig, filename, Tight=True, fig_dir=FIG_DIR):
    png_dir = fig_dir / "pngs"
    png_dir.mkdir(parents=True, exist_ok=True)

    for ax in fig.axes:
        if (ax.get_yscale() == "log"):  # label minor ticks when the axis spans only a decade or two
            ax.yaxis.set_minor_formatter(
                LogFormatterSciNotation(minor_thresholds=(2, 0.4))
            )
    if Tight:
        fig.tight_layout()
    fig.savefig(fig_dir / filename, bbox_inches="tight")  # include the legend
    fig.savefig(
        png_dir / filename.replace(".pdf", ".png"), dpi=150, bbox_inches="tight"
    )
    print(f"Saved: {fig_dir / filename}")
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
    below_axes = offset_copy(ax.transAxes, fig=fig, y=-30, units="points")

    ncol = min(ncol, len(handles))
    while True:
        legend = ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            bbox_transform=below_axes,
            ncol=ncol,
            frameon=True,
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

def scientific_label(x):
    if x == 0:
        return "0"
    exponent = int(np.round(np.log10(x)))
    coefficient = x / 10**exponent

    if np.isclose(coefficient, 1):
        return rf"$10^{{{exponent}}}$"

    return rf"${coefficient:g}\times10^{{{exponent}}}$"

