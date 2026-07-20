"""Ground-truth importance measures (gradient, ablation, optional Fisher).

Protocol contract:
- ``compute(model, probe_loader, criterion, edge_sample_mask=...)``
- Must **never** call ``LocalSignal.update`` (snapshot isolation).
"""

from neuronca.importance.ablation import (
    DEFAULT_EDGE_MASK_SEED_OFFSET,
    DEFAULT_EDGE_SAMPLE,
    DEFAULT_HIDDEN_LAYERS,
    AblationImportance,
    build_edge_ablation_mask,
    default_edge_mask_path,
    edge_mask_rng_seed,
    layer_id_hash,
    load_edge_ablation_mask,
    save_edge_ablation_mask,
)
from neuronca.importance.base import (
    GroundTruthImportance,
    ImportanceResult,
    iter_linear_layers,
    mean_probe_loss,
)
from neuronca.importance.fisher import FisherImportance, compute_fisher
from neuronca.importance.gradient_saliency import GradientSaliency, compute_grad_abs

__all__ = [
    "AblationImportance",
    "DEFAULT_EDGE_MASK_SEED_OFFSET",
    "DEFAULT_EDGE_SAMPLE",
    "DEFAULT_HIDDEN_LAYERS",
    "FisherImportance",
    "GradientSaliency",
    "GroundTruthImportance",
    "ImportanceResult",
    "build_edge_ablation_mask",
    "compute_fisher",
    "compute_grad_abs",
    "default_edge_mask_path",
    "edge_mask_rng_seed",
    "iter_linear_layers",
    "layer_id_hash",
    "load_edge_ablation_mask",
    "mean_probe_loss",
    "save_edge_ablation_mask",
]
