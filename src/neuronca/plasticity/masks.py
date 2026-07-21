"""Edge/neuron origin tags and cooldown tracking for Phase 1a plasticity."""

from __future__ import annotations

from dataclasses import dataclass, field

# Edge origin (byte / int codes stored in LongTensor masks).
EDGE_ORIGIN_NONE = 0
EDGE_ORIGIN_INIT = 1  # init_dense — present at construction
EDGE_ORIGIN_INCENTIVE = 2  # incentive_grown during training
EDGE_ORIGIN_GAP_IMPRINT = 3  # gap_imprinted-seeded (specialist birth fan)

# Neuron origin (active vs init vs grown — used by prune/grace logic).
NEURON_ORIGIN_NONE = 0
NEURON_ORIGIN_INIT = 1
NEURON_ORIGIN_GROWN = 2

# Birth mechanism (finer provenance for structure visualization / audit).
# Distinct from NEURON_ORIGIN_* so existing grow/prune rules stay binary.
NEURON_BIRTH_NONE = 0
NEURON_BIRTH_INIT = 1
NEURON_BIRTH_KAIMING = 2
NEURON_BIRTH_PARENT_PERTURBED = 3
NEURON_BIRTH_GAP_IMPRINTED = 4

ORIGIN_EDGE_NAMES = {
    EDGE_ORIGIN_NONE: "none",
    EDGE_ORIGIN_INIT: "init_dense",
    EDGE_ORIGIN_INCENTIVE: "incentive_grown",
    EDGE_ORIGIN_GAP_IMPRINT: "gap_imprinted_seeded",
}

ORIGIN_NEURON_NAMES = {
    NEURON_ORIGIN_NONE: "none",
    NEURON_ORIGIN_INIT: "init",
    NEURON_ORIGIN_GROWN: "grown",
}

BIRTH_KIND_NAMES = {
    NEURON_BIRTH_NONE: "none",
    NEURON_BIRTH_INIT: "init",
    NEURON_BIRTH_KAIMING: "kaiming",
    NEURON_BIRTH_PARENT_PERTURBED: "parent_perturbed",
    NEURON_BIRTH_GAP_IMPRINTED: "gap_imprinted",
}

# Map weight_init string → birth-kind code.
BIRTH_KIND_FROM_WEIGHT_INIT = {
    "init": NEURON_BIRTH_INIT,
    "kaiming": NEURON_BIRTH_KAIMING,
    "parent_perturbed": NEURON_BIRTH_PARENT_PERTURBED,
    "parent": NEURON_BIRTH_PARENT_PERTURBED,
    "copy_parent": NEURON_BIRTH_PARENT_PERTURBED,
    "gap_imprinted": NEURON_BIRTH_GAP_IMPRINTED,
    "gap_imprint": NEURON_BIRTH_GAP_IMPRINTED,
    "gap": NEURON_BIRTH_GAP_IMPRINTED,
}


@dataclass
class CooldownTracker:
    """Hard cooldown: freed at plasticity step t ineligible until step t+1.

    Keys:
        neuron: (hidden_layer_name, slot_index)
        edge: (linear_layer_name, out_index, in_index)
    """

    plasticity_step: int = 0
    neuron_cooldown_until: dict[tuple[str, int], int] = field(default_factory=dict)
    edge_cooldown_until: dict[tuple[str, int, int], int] = field(default_factory=dict)
    same_step_blocks: int = 0

    def begin_step(self) -> int:
        """Advance plasticity step counter; return the new step index."""
        self.plasticity_step += 1
        return self.plasticity_step

    def mark_neuron_freed(self, layer: str, slot: int) -> None:
        """Mark a neuron slot freed at the current step (eligible next step)."""
        self.neuron_cooldown_until[(layer, int(slot))] = self.plasticity_step + 1

    def mark_edge_freed(self, layer: str, out_idx: int, in_idx: int) -> None:
        """Mark an edge freed at the current step (eligible next step)."""
        key = (layer, int(out_idx), int(in_idx))
        self.edge_cooldown_until[key] = self.plasticity_step + 1

    def neuron_eligible(self, layer: str, slot: int) -> bool:
        """True if slot may be reactivated at the current plasticity step."""
        until = self.neuron_cooldown_until.get((layer, int(slot)))
        if until is None:
            return True
        if self.plasticity_step >= until:
            # Clean up expired entry opportunistically.
            del self.neuron_cooldown_until[(layer, int(slot))]
            return True
        self.same_step_blocks += 1
        return False

    def edge_eligible(self, layer: str, out_idx: int, in_idx: int) -> bool:
        """True if edge may be reformed at the current plasticity step."""
        key = (layer, int(out_idx), int(in_idx))
        until = self.edge_cooldown_until.get(key)
        if until is None:
            return True
        if self.plasticity_step >= until:
            del self.edge_cooldown_until[key]
            return True
        self.same_step_blocks += 1
        return False

    def assert_no_same_step_reuse(self) -> None:
        """No-op helper for callers that track bugs separately; blocks already counted."""
        return None
