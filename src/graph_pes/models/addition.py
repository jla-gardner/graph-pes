import torch
from torch import Tensor, nn

from graph_pes.core import GraphPESModel
from graph_pes.graphs import AtomicGraph, LabelledBatch
from graph_pes.util import uniform_repr


class AdditionModel(GraphPESModel):
    """
    A wrapper that makes predictions as the sum of the predictions
    of its constituent models.

    Parameters
    ----------
    models
        the models to sum.

    Examples
    --------
    Create a model with explicit two-body and multi-body terms:

    .. code-block:: python

        from graph_pes.models.zoo import LennardJones, SchNet
        from graph_pes.core import AdditionModel

        # create a model that sums two models
        # equivalent to LennardJones() + SchNet()
        model = AdditionModel([LennardJones(), SchNet()])
    """

    def __init__(self, **models: GraphPESModel):
        super().__init__()
        self.models = nn.ModuleDict(models)

    def predict_local_energies(self, graph: AtomicGraph) -> Tensor:
        predictions = torch.stack(
            [
                model.predict_local_energies(graph).squeeze()
                for model in self.models.values()
            ]
        )  # (atoms, models)
        return torch.sum(predictions, dim=0)  # (atoms,) sum over models

    def model_specific_pre_fit(self, graphs: LabelledBatch) -> None:
        for model in self.models.values():
            model.model_specific_pre_fit(graphs)

    def __repr__(self):
        return uniform_repr(
            self.__class__.__name__,
            **self.models,
            stringify=True,
            max_width=80,
        )
