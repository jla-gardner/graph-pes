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
from torch import Tensor

from ..atomic_graph import AtomicGraph, PropertyKey, to_batch
from ..graph_pes_model import GraphPESModel

Transform = Callable[[Tensor, AtomicGraph], Tensor]
r"""
Transforms map a property, :math:`x`, to a target property, :math:`y`,
conditioned on an :class:`~graph_pes.AtomicGraph`, :math:`\mathcal{G}`:

.. math::

    T: (x; \mathcal{G}) \mapsto y
"""


def identity(x: Tensor, graph: AtomicGraph) -> Tensor:
    return x


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


def parity_plot(
    model: GraphPESModel,
    graphs: AtomicGraph | Sequence[AtomicGraph],
    property: PropertyKey = "energy",
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
        The property to plot, e.g. :code:`"energy"`.
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

        parity_plot(model, train, "energy")

    .. image:: Cu-LJ-default-parity.svg
        :align: center

    Custom settings, as seen in
    :doc:`this example notebook <../quickstart/custom-training-loop>`:

    .. code-block:: python

        from graph_pes.atomic_graph import DividePerAtom
        from graph_pes.util import Keys

        for name, data, colour in zip(
            ["Train", "Test"],
            [train, test],
            ["royalblue", "crimson"],
        ):
            parity_plot(
                model,
                data,
                "energy",
                transform=DividePerAtom(),
                units="eV / atom",
                label=name,
                c=colour,
            )

        plt.legend(loc="upper left", fancybox=False);

    .. image:: ../quickstart/parity-plot.svg
        :align: center
    """
    # deal with defaults
    transform = transform or identity

    # get the predictions and labels
    if not isinstance(graphs, AtomicGraph):
        graphs = to_batch(graphs)

    raw_ground_truth = graphs.properties.get(property, None)
    if raw_ground_truth is None:
        raise ValueError(f"Property {property} not in graphs")
    ground_truth = transform(raw_ground_truth, graphs).detach()

    raw_pred = model.predict(graphs, properties=[property]).get(property, None)
    assert raw_pred is not None
    predictions = transform(raw_pred, graphs).detach()

    # plot
    ax: plt.Axes = ax or plt.gca()

    default_kwargs = dict(lw=0, clip_on=False)
    scatter_kwargs = {**default_kwargs, **scatter_kwargs}
    ax.scatter(ground_truth, predictions, **scatter_kwargs)

    # get a point guaranteed to be on the plot
    z = ground_truth.view(-1)[0].item()
    ax.axline((z, z), slope=1, c="k", ls="--", lw=1)

    # aesthetics
    axis_label = (
        f"{property.capitalize()} ({units})" if units else property.capitalize()
    )
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
    model: GraphPESModel,
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

    .. code-block:: python

        from graph_pes.utils.analysis import dimer_curve
        from graph_pes.models import LennardJones

        dimer_curve(LennardJones(sigma=1.3, epsilon=0.5), system="OH", units="eV")

    .. image:: dimer-curve.svg
        :align: center
    """  # noqa: E501

    trial_atoms = Atoms(system)
    if len(trial_atoms) != 2:
        system = system + "2"

    rs = np.linspace(rmin, rmax, 200)
    dimers = [Atoms(system, positions=[[0, 0, 0], [r, 0, 0]]) for r in rs]
    graphs = [AtomicGraph.from_ase(d, cutoff=rmax + 0.1) for d in dimers]
    batch = to_batch(graphs)

    with torch.no_grad():
        energy = model.predict_energy(batch).numpy()

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
