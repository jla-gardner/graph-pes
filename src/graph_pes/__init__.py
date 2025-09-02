import warnings

import torch

from graph_pes.atomic_graph import AtomicGraph
from graph_pes.graph_pes_model import GraphPESModel

# hide the annoying FutureWarning from e3nn
warnings.filterwarnings("ignore", category=FutureWarning, module="e3nn")

# fix e3nns torch.load without weights_only
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([slice])

__all__ = ["AtomicGraph", "GraphPESModel"]
__version__ = "0.2.0"
