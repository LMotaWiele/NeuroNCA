"""Phase 1a structural plasticity: local homeostatic grow/prune (no topology optimizer).

Import submodules directly for heavy symbols to avoid circular imports with
``neuronca.models.dynamic_mlp``:

    from neuronca.plasticity.orchestrator import PlasticityOrchestrator
    from neuronca.plasticity.rules import PlasticityConfig
"""

from neuronca.plasticity.masks import (
    EDGE_ORIGIN_GAP_IMPRINT,
    EDGE_ORIGIN_INCENTIVE,
    EDGE_ORIGIN_INIT,
    EDGE_ORIGIN_NONE,
    NEURON_ORIGIN_GROWN,
    NEURON_ORIGIN_INIT,
    NEURON_ORIGIN_NONE,
)
from neuronca.plasticity.optimizer_reset import reset_optimizer_state_for_indices

__all__ = [
    "EDGE_ORIGIN_NONE",
    "EDGE_ORIGIN_INIT",
    "EDGE_ORIGIN_INCENTIVE",
    "EDGE_ORIGIN_GAP_IMPRINT",
    "NEURON_ORIGIN_NONE",
    "NEURON_ORIGIN_INIT",
    "NEURON_ORIGIN_GROWN",
    "reset_optimizer_state_for_indices",
]
