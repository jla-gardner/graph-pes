from __future__ import annotations

from typing import Callable, Sequence

import matplotlib.axes
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import Atoms
from cycler import cycler
from matplotlib.ticker import MaxNLocator

from .core import ConservativePESModel, get_predictions
from .data.io import to_atomic_graph
from .graphs import AtomicGraph, AtomicGraphBatch, keys
from .graphs.operations import to_batch
from .transform import Transform, identity

_my_style = {
    "figure.figsize": (3.5, 3),
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
    "legend.fancybox": False,
    "savefig.transparent": False,
    "axes.prop_cycle": cycler(
        "color",
        [
            "#4385be",
            "#ce5d97",
            "#da702c",
            "#8b7ec8",
            "#879a39",
            "#d14d41",
            "#d14d41",
            "#d0a215",
            "#3aa99f",
        ],
    ),
}

plt.rcParams.update(_my_style)


def move_axes(ax: matplotlib.axes.Axes | None = None):  # type: ignore
    """
    Move the axes to the center of the figure
    """
    ax: plt.Axes = ax or plt.gca()
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))


# TODO per-property default transforms
# energy: per-atom with arbitrary shift
def parity_plot(
    model: Callable[[AtomicGraph], torch.Tensor],
    graphs: AtomicGraphBatch | Sequence[AtomicGraph],
    property: keys.LabelKey = keys.ENERGY,
    property_label: str | None = None,
    transform: Transform | None = None,
    units: str | None = None,
    ax: matplotlib.axes.Axes | None = None,  # type: ignore
    **scatter_kwargs,
):
    r"""
    A nicely formatted parity plot of model predictions vs ground truth
    for the given :code:`property`.

    Parameters
    ----------
    model
        The model to for generating predictions.
    graphs
        The graphs to make predictions on.
    property
        The property to plot, e.g. :code:`keys.ENERGY`.
    property_label
        The string that the property is indexed by on the graphs. If not
        provided, defaults to the value of :code:`property`, e.g.
        :code:`keys.ENERGY` :math:`\rightarrow` :code:`"energy"`.
    transform
        The transform to apply to the predictions and labels before plotting.
        If not provided, no transform is applied.
    units
        The units of the property, for labelling the axes. If not provided, no
        units are used.
    ax
        The axes to plot on. If not provided, the current axes are used.
    scatter_kwargs
        Keyword arguments to pass to :code:`plt.scatter`.

    Examples
    --------
    Default settings (no units, transforms or custom scatter keywords):

    .. code-block:: python

        parity_plot(model, train, keys.ENERGY)

    .. image:: notebooks/Cu-LJ-default-parity.svg
        :align: center

    Custom settings, as seen in
    :doc:`this example notebook <notebooks/example>`:

    .. code-block:: python

        from graph_pes.transform import DividePerAtom
        from graph_pes.util import Keys

        for name, data, colour in zip(
            ["Train", "Test"],
            [train, test],
            ["royalblue", "crimson"],
        ):
            parity_plot(
                model,
                data,
                keys.ENERGY,
                transform=DividePerAtom(),
                units="eV / atom",
                label=name,
                c=colour,
            )

        plt.legend(loc="upper left", fancybox=False);

    .. image:: notebooks/Cu-LJ-parity.svg
        :align: center
    """
    # deal with defaults
    transform = transform or identity
    if property_label is None:
        property_label = property

    # get the predictions and labels
    if isinstance(graphs, Sequence):
        graphs = to_batch(graphs)

    ground_truth = transform(graphs[property_label], graphs).detach()
    pred = get_predictions(model, graphs, property=property)
    predictions = transform(pred, graphs).detach()

    # plot
    ax: plt.Axes = ax or plt.gca()

    default_kwargs = dict(lw=0, clip_on=False)
    scatter_kwargs = {**default_kwargs, **scatter_kwargs}
    ax.scatter(ground_truth, predictions, **scatter_kwargs)

    # get a point guaranteed to be on the plot
    z = ground_truth.view(-1)[0].item()
    ax.axline((z, z), slope=1, c="k", ls="--", lw=1)

    # aesthetics
    axis_label = f"{property_label}  ({units})" if units else property_label
    ax.set_xlabel(f"True {axis_label}")
    ax.set_ylabel(f"Predicted {axis_label}")
    ax.set_aspect("equal", "datalim")
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xlim(min(x0, y0), max(x1, y1))
    ax.set_ylim(min(x0, y0), max(x1, y1))
    move_axes(ax)

    # 5 ticks each
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))


def dimer_curve(
    model: ConservativePESModel,
    system: str,
    units: str | None = None,
    set_to_zero: bool = True,
    rmin: float = 0.9,
    rmax: float = 5.0,
    ax: matplotlib.axes.Axes | None = None,  # type: ignore
    **plot_kwargs,
) -> matplotlib.lines.Line2D:
    r"""
    A nicely formatted dimer curve plot for the given :code:`system`.

    Parameters
    ----------
    model
        The model to for generating predictions.
    system
        The dimer system. Should be one of: a single element, e.g. :code:`"Cu"`,
        or a pair of elements, e.g. :code:`"CuO"`.
    units
        The units of the energy, for labelling the axes. If not provided, no
        units are used.
    set_to_zero
        Whether to set the energy of the dimer at :code:`rmax` to be zero.
    rmin
        The minimum seperation to consider.
    rmax
        The maximum seperation to consider.
    ax
        The axes to plot on. If not provided, the current axes are used.
    plot_kwargs
        Keyword arguments to pass to :code:`plt.plot`.

    Examples
    --------
    See :doc:`this example notebook <notebooks/example>`:

    .. code-block:: python

        dimer_curve(model, "Cu", units="eV", label="Final", c="C1")

    .. image:: notebooks/Cu-LJ-dimer.svg
        :align: center
    """

    trial_atoms = Atoms(system)
    if len(trial_atoms) != 2:
        system = system + "2"

    rs = np.linspace(rmin, rmax, 200)
    dimers = [Atoms(system, positions=[[0, 0, 0], [r, 0, 0]]) for r in rs]
    graphs = [to_atomic_graph(d, cutoff=rmax + 0.1) for d in dimers]
    batch = to_batch(graphs)

    with torch.no_grad():
        energy = model(batch).numpy()

    if set_to_zero:
        energy -= energy[-1]

    ax: plt.Axes = ax or plt.gca()

    default_kwargs = dict(lw=1, c="k")
    plot_kwargs = {**default_kwargs, **plot_kwargs}
    line = ax.plot(rs, energy, **plot_kwargs)[0]
    assert isinstance(line, matplotlib.lines.Line2D)

    limiting_energy = energy[-1]
    if (energy[:-1] < limiting_energy).any():
        well_depth = limiting_energy - energy[:-1].min()
    else:
        well_depth = 0.1
    bottom = limiting_energy - well_depth * 1.1
    top = limiting_energy + well_depth * 1.1
    ax.set_ylim(bottom, top)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel(f"Dimer Energy ({units})" if units else "Dimer Energy")

    first_in_view = np.where(energy < top)[0][0]
    ax.set_xlim(rs[first_in_view].item() - 0.2, rs[-1] + 0.2)
    move_axes(ax)

    # 5 ticks each
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))

    return line
