import string
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib_scalebar
import scipy.interpolate


def transect(pts, ppf, paui, duri, grid=True):
    """Interpolate phonotaxis response on a transect grid.

    Args:
        pts (array-like): Sample points for interpolation.
        ppf (array-like): Response values at sample points.
        paui (array-like): Pause values or grid points.
        duri (array-like): Duration values or grid points.
        grid (bool, optional): Whether to meshgrid paui/duri before interpolation.

    Returns:
        numpy.ndarray: Flattened interpolated transect values.
    """
    if grid:
        duri, paui = np.meshgrid(duri, paui)
    transect = scipy.interpolate.griddata(
        pts, ppf.flatten(), (duri, paui), method="linear"
    )
    transect = transect.flatten()
    return transect


def label_axes(fig=None, labels=None, loc=None, **kwargs):
    """Label each axes in a figure with a running set of labels.

    Args:
        fig (matplotlib.figure.Figure, optional): Figure to label. Defaults to
            current figure.
        labels (iterable[str] | None, optional): Labels to cycle through.
            Defaults to uppercase letters.
        loc (tuple[float, float] | None, optional): Label location in
            axes-fraction units.
        **kwargs: Passed through to `Axes.annotate`.
    """
    if fig is None:
        fig = plt.gcf()

    if labels is None:
        labels = string.ascii_uppercase

    if "size" not in kwargs:
        kwargs["size"] = 24

    if "weight" not in kwargs:
        kwargs["fontweight"] = "heavy"

    # re-use labels rather than stop labeling
    labels = cycle(labels)
    if loc is None:
        loc = (-0.2, 0.9)

    for ax in fig.axes:
        if "colorbar" in ax.axes.get_label():
            continue
        lab = next(labels)
        ax.annotate(lab, xy=loc, xycoords="axes fraction", **kwargs)


def ppf(dur, pau, ppf, ax=None, colorbar=True):
    """Plot a phonotaxis performance field.

    Args:
        dur (array-like): Pulse duration grid or values.
        pau (array-like): Pause grid or values.
        ppf (array-like): Phonotaxis response field.
        ax (matplotlib.axes.Axes | None, optional): Axes to plot into.
        colorbar (bool, optional): Whether to add a colorbar.
    """
    if ax is not None:
        plt.sca(ax)

    ax = plt.gca()
    plt.pcolor(dur, pau, ppf, cmap="Greys")
    plt.axis("square")
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.xlabel("Pulse [ms]")
    plt.ylabel("Pause [ms]")
    plt.xticks(np.arange(0, 20.1, 5))
    plt.yticks(np.arange(0, 20.1, 5))
    if colorbar:
        cax = inset_axes(plt.gca(), width="5%", height="20%", loc=1)
        cax = plt.colorbar(cax=cax, label="Phonotaxis", fraction=0.02, pad=0.03)
        cax.set_ticks([0, np.around(np.nanmax(ppf), decimals=2)])
    plt.sca(ax)


def pulse(x, c="k", alpha=0.5, dt=1, offset=0):
    """Plot a pulse waveform as a filled step-like shape.

    Args:
        x (array-like): Pulse samples; modified in-place for plotting.
        c (str, optional): Line and fill color.
        alpha (float, optional): Fill transparency.
        dt (float, optional): Sample rate scaling for the time axis.
        offset (float, optional): Baseline offset to add to the pulse.
    """
    T = np.arange(0, len(x)) / dt
    x[-1] = x[0]
    x += offset
    x[0] = offset
    x[-1] = offset
    plt.fill(T, x, c=c, alpha=alpha)
    plt.plot(T, x, c=c)
    plt.plot(T, np.zeros_like(x) + x[0], c=c)
    plt.axhline(offset, c="k", linewidth=0.5)


def despine(which="tr", axis=None):
    """Hide specified axis spines and associated ticks.

    Args:
        which (str, optional): Combination of 't', 'b', 'l', 'r' to hide.
        axis (matplotlib.axes.Axes | None, optional): Axes to modify.
    """
    sides = {"t": "top", "b": "bottom", "l": "left", "r": "right"}

    if axis is None:
        axis = plt.gca()

    # Hide the spines
    for side in which:
        axis.spines[sides[side]].set_visible(False)

    # Hide the tick marks and labels
    if "r" in which:
        axis.yaxis.set_ticks_position("left")

    if "t" in which:
        axis.xaxis.set_ticks_position("bottom")

    if "l" in which:
        axis.yaxis.set_ticks([])

    if "b" in which:
        axis.xaxis.set_ticks([])


def scalebar(
    length,
    dx=1,
    units="",
    label=None,
    axis=None,
    location="lower right",
    frameon=False,
    **kwargs,
):
    """Add a scalebar artist to an axis.

    Args:
        length (float): Length of the scalebar in axis tick units.
        dx (int, optional): Scale factor for length. A value of 10 makes a
            length of 1.0 span 10 ticks. Defaults to 1.
        units (str, optional): Unit label (e.g. "milliseconds").
        label (str, optional): Title for the scalebar.
        axis (matplotlib.axes.Axes | None, optional): Axes to add the scalebar
            to. Defaults to the current axis.
        location (str, optional): Axes location descriptor.
        frameon (bool, optional): Whether to draw a background frame.
        **kwargs: Passed through to `matplotlib_scalebar.ScaleBar`.

    Returns:
        matplotlib_scalebar.scalebar.ScaleBar: The added scalebar artist.
    """

    if axis is None:
        axis = plt.gca()

    if "dimension" not in kwargs:
        kwargs["dimension"] = matplotlib_scalebar.dimension._Dimension(units)

    scalebar = ScaleBar(
        dx=dx,
        units=units,
        label=label,
        fixed_value=length,
        location=location,
        frameon=frameon,
        **kwargs,
    )
    axis.add_artist(scalebar)
    return scalebar


def plot_boundaries(ax=None):
    """Plot the chirp boundaries on duration/pause axes measured in seconds."""
    if ax is None:
        ax = plt.gca()
    ax.plot([0, .6], [.6, 0], c="royalblue", label="Period <600 ms")
    ax.plot([0, 1], [0, .25], c="seagreen", label="Duty cycle <0.8")
    ax.plot([.04, .04], [0, 1], c="mediumorchid", label="Pulses >1")


def transect_est(data, filter_key, filter_val, plot_x_key, plot_y_key="rXY"):
    """Estimate a smoothed transect slice from filtered data.

    Args:
        data (dict[str, numpy.ndarray]): Dataset with keyed arrays.
        filter_key (str): Key to filter on.
        filter_val (tuple[float, float]): (min, max) filter bounds.
        plot_key (str): Key for the x-axis values.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: x values, raw y
        maxima, and smoothed y values.
    """
    idx = np.where(
        np.logical_and(
            data[filter_key] > filter_val[0], data[filter_key] < filter_val[1]
        )
    )[0]
    x, y = data[plot_x_key][idx], data[plot_y_key][idx]
    # Fix duplicate x vals by avg y over all x with same val
    xm = np.unique(x)
    y = np.array([np.nanmax(y[x == xi]) for xi in xm])
    x = xm
    # sort
    sort_idx = np.argsort(x)
    x, y = x[sort_idx], y[sort_idx]
    # smooth
    ypad = np.pad(y, (1, 1), mode="edge")  # pad repeat edge values
    # ypad = np.pad(y, (1, 1), mode="constant", constant_values=np.nan)  # pad repeat edge values
    ys = np.convolve(ypad, np.ones(3) / 3, mode="valid")  # 3-val running avg
    return x, y, ys


def grpmax(ax=None, plt_kwargs={"c": "k"}):
    """Plot and return grouped maxima across all lines in an axis.

    Args:
        ax (matplotlib.axes.Axes | None, optional): Axes to inspect.
        plt_kwargs (dict, optional): Keyword args passed to `Axes.plot`.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Unique x values and max y values.
    """
    ax = plt.gca()
    xs, ys = [], []
    for line in ax.lines:
        x, y = line.get_data()
        xs.append(x)
        ys.append(y)

    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    xm = np.unique(x_all)
    ym = np.array([np.nanmax(y_all[x_all == xi]) for xi in xm])

    ax.plot(xm, ym, **plt_kwargs)

    return xm, ym


def nmax(x, value=None):
    if value is None:
        value = np.nanmax(x)
    return x / value
