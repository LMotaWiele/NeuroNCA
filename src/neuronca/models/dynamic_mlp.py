"""Capacity-reserved masked MLP with plastic hidden neurons and adjacent edges.

Phase 1a: fixed four-stage layout (input → H1 → H2 → H3 → output). Neuron counts
and inter-layer edges change via masks; weight tensors keep fixed capacity.
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

from neuronca.plasticity.coords import (
    assign_hidden_coord,
    input_coords_28x28,
    output_coords,
)
from neuronca.plasticity.masks import (
    BIRTH_KIND_FROM_WEIGHT_INIT,
    BIRTH_KIND_NAMES,
    EDGE_ORIGIN_GAP_IMPRINT,
    EDGE_ORIGIN_INCENTIVE,
    EDGE_ORIGIN_INIT,
    EDGE_ORIGIN_NONE,
    NEURON_BIRTH_INIT,
    NEURON_BIRTH_NONE,
    NEURON_ORIGIN_GROWN,
    NEURON_ORIGIN_INIT,
    NEURON_ORIGIN_NONE,
    CooldownTracker,
)
from neuronca.plasticity.optimizer_reset import (
    reset_linear_edge_optimizer_state,
    reset_linear_neuron_optimizer_state,
    reset_optimizer_state_for_indices,
)

# Hidden layer names (active masks / coords) and Linear names (edge masks).
HIDDEN_LAYERS = ("h1", "h2", "h3")
LINEAR_LAYERS = ("fc1", "fc2", "fc3", "fc_out")

# Map linear layer → (pre_population, post_population)
# Populations: "in", "h1", "h2", "h3", "out"
_LINEAR_ENDPOINTS = {
    "fc1": ("in", "h1"),
    "fc2": ("h1", "h2"),
    "fc3": ("h2", "h3"),
    "fc_out": ("h3", "out"),
}


class DynamicMLPReasoningNet(nn.Module):
    """Masked-capacity MLP with Phase 0-compatible activation keys.

    Capacity tensors are fixed at construction. Structure changes activate or
    deactivate slots and edges. Coordinates are birth-identity only (fresh draw
    on reactivate). Optimizer state must be scrubbed on (re)activation via the
    grow APIs when an optimizer is provided.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_init: tuple[int, ...] = (256, 128, 64),
        hidden_max: tuple[int, ...] = (512, 256, 128),
        num_classes: int = 10,
        *,
        coord_seed: int | None = 0,
    ) -> None:
        super().__init__()
        if len(hidden_init) != 3 or len(hidden_max) != 3:
            raise ValueError("hidden_init and hidden_max must each have length 3")
        for i, (hi, hm) in enumerate(zip(hidden_init, hidden_max)):
            if hi <= 0 or hm < hi:
                raise ValueError(
                    f"hidden layer {i}: need 0 < init={hi} <= max={hm}"
                )
        if input_dim != 784:
            # Coords assume 28×28 layout for MNIST; keep explicit for Phase 1a.
            pass

        self.input_dim = int(input_dim)
        self.hidden_init = tuple(int(x) for x in hidden_init)
        self.hidden_max = tuple(int(x) for x in hidden_max)
        self.num_classes = int(num_classes)
        self.hidden = self.hidden_init  # Phase 0 alias: initial active sizes

        h1_max, h2_max, h3_max = self.hidden_max
        self.fc1 = nn.Linear(self.input_dim, h1_max)
        self.fc2 = nn.Linear(h1_max, h2_max)
        self.fc3 = nn.Linear(h2_max, h3_max)
        self.fc_out = nn.Linear(h3_max, self.num_classes)

        # Active neuron masks (buffers, not parameters).
        self.register_buffer("active_h1", torch.zeros(h1_max, dtype=torch.bool))
        self.register_buffer("active_h2", torch.zeros(h2_max, dtype=torch.bool))
        self.register_buffer("active_h3", torch.zeros(h3_max, dtype=torch.bool))

        self.register_buffer(
            "neuron_origin_h1",
            torch.zeros(h1_max, dtype=torch.uint8),
        )
        self.register_buffer(
            "neuron_origin_h2",
            torch.zeros(h2_max, dtype=torch.uint8),
        )
        self.register_buffer(
            "neuron_origin_h3",
            torch.zeros(h3_max, dtype=torch.uint8),
        )
        # Birth mechanism provenance (init / kaiming / parent_perturbed / gap_imprinted).
        self.register_buffer(
            "birth_kind_h1", torch.zeros(h1_max, dtype=torch.uint8)
        )
        self.register_buffer(
            "birth_kind_h2", torch.zeros(h2_max, dtype=torch.uint8)
        )
        self.register_buffer(
            "birth_kind_h3", torch.zeros(h3_max, dtype=torch.uint8)
        )
        # Growth-trigger / parent locus for structured births (-1 = none).
        self.register_buffer(
            "parent_slot_h1", torch.full((h1_max,), -1, dtype=torch.long)
        )
        self.register_buffer(
            "parent_slot_h2", torch.full((h2_max,), -1, dtype=torch.long)
        )
        self.register_buffer(
            "parent_slot_h3", torch.full((h3_max,), -1, dtype=torch.long)
        )
        # Plasticity step index at birth (-1 = inactive / never born).
        # Used for newborn grace before contraction prune (separate from cooldown).
        self.register_buffer("birth_step_h1", torch.full((h1_max,), -1, dtype=torch.long))
        self.register_buffer("birth_step_h2", torch.full((h2_max,), -1, dtype=torch.long))
        self.register_buffer("birth_step_h3", torch.full((h3_max,), -1, dtype=torch.long))
        # Temporary positive pre-activation bias for newborns (scaffold; not a Parameter).
        self.register_buffer("birth_bias_h1", torch.zeros(h1_max, dtype=torch.float32))
        self.register_buffer("birth_bias_h2", torch.zeros(h2_max, dtype=torch.float32))
        self.register_buffer("birth_bias_h3", torch.zeros(h3_max, dtype=torch.float32))
        # Initial bias magnitude at last birth (for linear decay schedule).
        self.register_buffer("birth_bias0_h1", torch.zeros(h1_max, dtype=torch.float32))
        self.register_buffer("birth_bias0_h2", torch.zeros(h2_max, dtype=torch.float32))
        self.register_buffer("birth_bias0_h3", torch.zeros(h3_max, dtype=torch.float32))

        # Coords: NaN for inactive / never assigned.
        self.register_buffer("coords_h1", torch.full((h1_max, 2), float("nan")))
        self.register_buffer("coords_h2", torch.full((h2_max, 2), float("nan")))
        self.register_buffer("coords_h3", torch.full((h3_max, 2), float("nan")))
        if self.input_dim == 784:
            coords_in = input_coords_28x28()
        else:
            xs = torch.linspace(0, 1, self.input_dim)
            coords_in = torch.stack([xs, torch.zeros(self.input_dim)], dim=-1)
        self.register_buffer("coords_in", coords_in)
        self.register_buffer("coords_out", output_coords(self.num_classes))

        # Edge presence + origin for each Linear: [out_cap, in_cap]
        for name, mod in self.named_linear_layers():
            self.register_buffer(
                f"edge_mask_{name}",
                torch.zeros(mod.out_features, mod.in_features, dtype=torch.bool),
            )
            self.register_buffer(
                f"edge_origin_{name}",
                torch.zeros(mod.out_features, mod.in_features, dtype=torch.uint8),
            )

        self.cooldown = CooldownTracker()
        self._coord_generator = torch.Generator()
        if coord_seed is not None:
            self._coord_generator.manual_seed(int(coord_seed))

        self._init_weights()
        self._init_topology()
        self.last_activations: dict[str, Tensor] | None = None

    # ------------------------------------------------------------------ setup
    def _init_weights(self) -> None:
        for _, module in self.named_linear_layers():
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _init_topology(self) -> None:
        """Activate initial hidden slots and fully connect adjacent active sets."""
        h1_i, h2_i, h3_i = self.hidden_init
        gen = self._coord_generator

        for slot in range(h1_i):
            self._birth_neuron("h1", slot, NEURON_ORIGIN_INIT, gen=gen)
        for slot in range(h2_i):
            self._birth_neuron("h2", slot, NEURON_ORIGIN_INIT, gen=gen)
        for slot in range(h3_i):
            self._birth_neuron("h3", slot, NEURON_ORIGIN_INIT, gen=gen)

        # Fully connect init-active populations (origin = init_dense).
        self._connect_dense("fc1", range(h1_i), range(self.input_dim), EDGE_ORIGIN_INIT)
        self._connect_dense("fc2", range(h2_i), range(h1_i), EDGE_ORIGIN_INIT)
        self._connect_dense("fc3", range(h3_i), range(h2_i), EDGE_ORIGIN_INIT)
        self._connect_dense(
            "fc_out", range(self.num_classes), range(h3_i), EDGE_ORIGIN_INIT
        )

    def _connect_dense(
        self,
        layer: str,
        outs: range | list[int],
        ins: range | list[int],
        origin: int,
    ) -> None:
        em = self.edge_mask(layer)
        eo = self.edge_origin(layer)
        for o in outs:
            for i in ins:
                em[o, i] = True
                eo[o, i] = origin

    def _birth_neuron(
        self,
        layer: str,
        slot: int,
        origin: int,
        *,
        gen: torch.Generator | None = None,
        coord: Tensor | None = None,
        birth_step: int | None = None,
        birth_kind: int | None = None,
        parent_slot: int | None = None,
    ) -> Tensor:
        active = self.active_mask(layer)
        if bool(active[slot]):
            raise ValueError(f"{layer}[{slot}] already active")
        active[slot] = True
        self.neuron_origin(layer)[slot] = origin
        if birth_kind is None:
            birth_kind = (
                NEURON_BIRTH_INIT if origin == NEURON_ORIGIN_INIT else NEURON_BIRTH_NONE
            )
        self.birth_kind(layer)[slot] = int(birth_kind)
        self.parent_slot(layer)[slot] = (
            int(parent_slot) if parent_slot is not None else -1
        )
        # Init topology: treat as age-infinite (birth_step=0 at step 0, or -2 sentinel
        # meaning "no grace"). Use 0 for init so age is large once plasticity runs.
        if birth_step is not None:
            self.birth_step(layer)[slot] = int(birth_step)
        elif origin == NEURON_ORIGIN_INIT:
            self.birth_step(layer)[slot] = 0
        else:
            self.birth_step(layer)[slot] = int(self.cooldown.plasticity_step)
        if coord is None:
            others = active.clone()
            others[slot] = False
            existing: Tensor | None
            if not others.any():
                existing = None
            else:
                existing = self.coords(layer)[others.nonzero(as_tuple=False).view(-1)]
                if existing.numel() and torch.isnan(existing).any():
                    valid = ~torch.isnan(existing).any(dim=-1)
                    existing = existing[valid] if valid.any() else None
            coord = assign_hidden_coord(
                generator=gen or self._coord_generator,
                device=self.coords(layer).device,
                existing=existing,
            )
        self.coords(layer)[slot] = coord.to(
            device=self.coords(layer).device, dtype=self.coords(layer).dtype
        )
        return self.coords(layer)[slot]

    # ------------------------------------------------------------------ accessors
    def named_linear_layers(self) -> list[tuple[str, nn.Linear]]:
        return [
            ("fc1", self.fc1),
            ("fc2", self.fc2),
            ("fc3", self.fc3),
            ("fc_out", self.fc_out),
        ]

    def layer_shapes(self) -> dict[str, tuple[int, int]]:
        """Capacity shapes (out, in) for signal buffer allocation."""
        return {
            name: (mod.out_features, mod.in_features)
            for name, mod in self.named_linear_layers()
        }

    def active_layer_shapes(self) -> dict[str, tuple[int, int]]:
        """Active-only (out_count, in_count) for diagnostics."""
        return {
            "fc1": (int(self.active_h1.sum()), self.input_dim),
            "fc2": (int(self.active_h2.sum()), int(self.active_h1.sum())),
            "fc3": (int(self.active_h3.sum()), int(self.active_h2.sum())),
            "fc_out": (self.num_classes, int(self.active_h3.sum())),
        }

    def active_mask(self, layer: str) -> Tensor:
        if layer == "h1":
            return self.active_h1
        if layer == "h2":
            return self.active_h2
        if layer == "h3":
            return self.active_h3
        raise KeyError(f"unknown hidden layer {layer!r}")

    def neuron_origin(self, layer: str) -> Tensor:
        if layer == "h1":
            return self.neuron_origin_h1
        if layer == "h2":
            return self.neuron_origin_h2
        if layer == "h3":
            return self.neuron_origin_h3
        raise KeyError(f"unknown hidden layer {layer!r}")

    def birth_kind(self, layer: str) -> Tensor:
        if layer == "h1":
            return self.birth_kind_h1
        if layer == "h2":
            return self.birth_kind_h2
        if layer == "h3":
            return self.birth_kind_h3
        raise KeyError(f"unknown hidden layer {layer!r}")

    def parent_slot(self, layer: str) -> Tensor:
        if layer == "h1":
            return self.parent_slot_h1
        if layer == "h2":
            return self.parent_slot_h2
        if layer == "h3":
            return self.parent_slot_h3
        raise KeyError(f"unknown hidden layer {layer!r}")

    def birth_step(self, layer: str) -> Tensor:
        if layer == "h1":
            return self.birth_step_h1
        if layer == "h2":
            return self.birth_step_h2
        if layer == "h3":
            return self.birth_step_h3
        raise KeyError(f"unknown hidden layer {layer!r}")

    def neuron_age_steps(self, layer: str, slot: int) -> int:
        """Plasticity steps since birth; -1 if inactive / never born."""
        b = int(self.birth_step(layer)[slot].item())
        if b < 0:
            return -1
        return int(self.cooldown.plasticity_step) - b

    def birth_bias(self, layer: str) -> Tensor:
        if layer == "h1":
            return self.birth_bias_h1
        if layer == "h2":
            return self.birth_bias_h2
        if layer == "h3":
            return self.birth_bias_h3
        raise KeyError(f"unknown hidden layer {layer!r}")

    def birth_bias0(self, layer: str) -> Tensor:
        if layer == "h1":
            return self.birth_bias0_h1
        if layer == "h2":
            return self.birth_bias0_h2
        if layer == "h3":
            return self.birth_bias0_h3
        raise KeyError(f"unknown hidden layer {layer!r}")

    def update_birth_biases(self, decay_steps: int) -> None:
        """Linearly decay temporary birth biases to 0 over ``decay_steps`` ages.

        age = plasticity_step - birth_step. bias = bias0 * max(0, 1 - age/decay_steps).
        """
        if decay_steps <= 0:
            for h in HIDDEN_LAYERS:
                self.birth_bias(h).zero_()
                self.birth_bias0(h).zero_()
            return
        step = int(self.cooldown.plasticity_step)
        for h in HIDDEN_LAYERS:
            birth = self.birth_step(h)
            b0 = self.birth_bias0(h)
            bb = self.birth_bias(h)
            active = self.active_mask(h)
            age = (step - birth).clamp(min=0).float()
            # Only slots with a recorded birth and nonzero schedule
            scale = (1.0 - age / float(decay_steps)).clamp(min=0.0)
            bb.copy_(torch.where(active & (birth >= 0), b0 * scale, torch.zeros_like(bb)))

    def coords(self, population: str) -> Tensor:
        if population == "in":
            return self.coords_in
        if population == "out":
            return self.coords_out
        if population == "h1":
            return self.coords_h1
        if population == "h2":
            return self.coords_h2
        if population == "h3":
            return self.coords_h3
        raise KeyError(f"unknown population {population!r}")

    def edge_mask(self, layer: str) -> Tensor:
        return getattr(self, f"edge_mask_{layer}")

    def edge_origin(self, layer: str) -> Tensor:
        return getattr(self, f"edge_origin_{layer}")

    def linear_module(self, layer: str) -> nn.Linear:
        return dict(self.named_linear_layers())[layer]

    def population_active(self, population: str) -> Tensor:
        """Boolean mask of active units; input/output always all-True."""
        if population == "in":
            return torch.ones(self.input_dim, dtype=torch.bool, device=self.coords_in.device)
        if population == "out":
            return torch.ones(
                self.num_classes, dtype=torch.bool, device=self.coords_out.device
            )
        return self.active_mask(population)

    def hidden_counts(self) -> dict[str, int]:
        return {h: int(self.active_mask(h).sum()) for h in HIDDEN_LAYERS}

    def edge_counts(self, *, by_origin: bool = False) -> dict:
        """Count present edges per linear layer (optionally by origin)."""
        out: dict = {}
        for name, _ in self.named_linear_layers():
            em = self.edge_mask(name)
            eo = self.edge_origin(name)
            if not by_origin:
                out[name] = int(em.sum())
            else:
                out[name] = {
                    "init_dense": int(((eo == EDGE_ORIGIN_INIT) & em).sum()),
                    "incentive_grown": int(((eo == EDGE_ORIGIN_INCENTIVE) & em).sum()),
                    "gap_imprinted_seeded": int(
                        ((eo == EDGE_ORIGIN_GAP_IMPRINT) & em).sum()
                    ),
                    "total": int(em.sum()),
                }
        return out

    def active_parameter_count(self) -> int:
        """Count weight/bias entries that participate in the active masked graph."""
        total = 0
        for name, mod in self.named_linear_layers():
            em = self.edge_mask(name)
            pre, post = _LINEAR_ENDPOINTS[name]
            pre_a = self.population_active(pre)
            post_a = self.population_active(post)
            used = em & post_a.unsqueeze(1) & pre_a.unsqueeze(0)
            total += int(used.sum())
            if mod.bias is not None:
                total += int(post_a.sum())
        return total

    def capacity_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------ masks / forward
    def effective_weight(self, layer: str) -> Tensor:
        """Weight × edge_mask × active row/col masks (autograd through weight)."""
        mod = self.linear_module(layer)
        pre, post = _LINEAR_ENDPOINTS[layer]
        em = self.edge_mask(layer).to(dtype=mod.weight.dtype)
        pre_a = self.population_active(pre).to(dtype=mod.weight.dtype)
        post_a = self.population_active(post).to(dtype=mod.weight.dtype)
        return mod.weight * em * post_a.unsqueeze(1) * pre_a.unsqueeze(0)

    def effective_bias(self, layer: str) -> Tensor | None:
        mod = self.linear_module(layer)
        _, post = _LINEAR_ENDPOINTS[layer]
        post_a = self.population_active(post)
        # Base Parameter bias (may be None) + temporary newborn birth bias on h*.
        if mod.bias is not None:
            base = mod.bias * post_a.to(dtype=mod.bias.dtype)
        else:
            base = torch.zeros(
                mod.out_features,
                device=mod.weight.device,
                dtype=mod.weight.dtype,
            )
        if post in HIDDEN_LAYERS:
            bb = self.birth_bias(post).to(device=base.device, dtype=base.dtype)
            base = base + bb * post_a.to(dtype=base.dtype)
            return base
        # Non-hidden (fc_out): only Parameter bias if present.
        if mod.bias is None:
            return None
        return base

    def forward(
        self,
        x: Tensor,
        return_activations: bool = False,
    ) -> Union[Tensor, tuple[Tensor, dict[str, Tensor]]]:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"expected last dim {self.input_dim}, got shape {tuple(x.shape)}"
            )

        act_in = x
        h1 = F.relu(
            F.linear(act_in, self.effective_weight("fc1"), self.effective_bias("fc1"))
        )
        h2 = F.relu(
            F.linear(h1, self.effective_weight("fc2"), self.effective_bias("fc2"))
        )
        h3 = F.relu(
            F.linear(h2, self.effective_weight("fc3"), self.effective_bias("fc3"))
        )
        logits = F.linear(
            h3, self.effective_weight("fc_out"), self.effective_bias("fc_out")
        )

        activations: dict[str, Tensor] = {
            "act_in": act_in,
            "act_fc1": h1,
            "act_fc2": h2,
            "act_fc3": h3,
            "act_fc_out": logits,
        }
        self.last_activations = {k: v.detach().clone() for k, v in activations.items()}
        if return_activations:
            return logits, activations
        return logits

    @torch.no_grad()
    def zero_inactive_grads(self) -> None:
        """Zero gradients on inactive rows/cols and absent edges after backward."""
        for name, mod in self.named_linear_layers():
            if mod.weight.grad is None:
                continue
            pre, post = _LINEAR_ENDPOINTS[name]
            em = self.edge_mask(name)
            pre_a = self.population_active(pre)
            post_a = self.population_active(post)
            used = em & post_a.unsqueeze(1) & pre_a.unsqueeze(0)
            mod.weight.grad.masked_fill_(~used, 0.0)
            if mod.bias is not None and mod.bias.grad is not None:
                mod.bias.grad.masked_fill_(~post_a, 0.0)

    # ------------------------------------------------------------------ structure API
    def begin_plasticity_step(self) -> int:
        """Advance cooldown clock; call once at the start of each orchestrator step."""
        return self.cooldown.begin_step()

    @torch.no_grad()
    def deactivate_neuron(self, layer: str, slot: int) -> dict:
        """Prune a hidden neuron: free slot, drop incident edges, mark cooldown."""
        active = self.active_mask(layer)
        if not bool(active[slot]):
            raise ValueError(f"{layer}[{slot}] is not active")
        origin = int(self.neuron_origin(layer)[slot].item())
        birth_kind = int(self.birth_kind(layer)[slot].item())
        parent = int(self.parent_slot(layer)[slot].item())
        bstep = int(self.birth_step(layer)[slot].item())
        # Capture last living coords before free (needed for structure viz death marks).
        coord = self.coords(layer)[slot].detach().float().cpu().clone()
        x = float(coord[0].item()) if torch.isfinite(coord).all() else float("nan")
        y = float(coord[1].item()) if torch.isfinite(coord).all() else float("nan")
        active[slot] = False
        self.neuron_origin(layer)[slot] = NEURON_ORIGIN_NONE
        self.birth_kind(layer)[slot] = NEURON_BIRTH_NONE
        self.parent_slot(layer)[slot] = -1
        self.birth_step(layer)[slot] = -1
        self.birth_bias(layer)[slot] = 0.0
        self.birth_bias0(layer)[slot] = 0.0
        self.coords(layer)[slot] = float("nan")

        edges_removed: list[tuple[str, int, int, int]] = []
        # Remove edges where this neuron is post (outgoing from previous layer).
        for lin, (pre, post) in _LINEAR_ENDPOINTS.items():
            em = self.edge_mask(lin)
            eo = self.edge_origin(lin)
            if post == layer:
                ins = em[slot].nonzero(as_tuple=False).view(-1).tolist()
                for j in ins:
                    o = int(eo[slot, j].item())
                    em[slot, j] = False
                    eo[slot, j] = EDGE_ORIGIN_NONE
                    self.cooldown.mark_edge_freed(lin, slot, j)
                    edges_removed.append((lin, slot, j, o))
            if pre == layer:
                outs = em[:, slot].nonzero(as_tuple=False).view(-1).tolist()
                for i in outs:
                    o = int(eo[i, slot].item())
                    em[i, slot] = False
                    eo[i, slot] = EDGE_ORIGIN_NONE
                    self.cooldown.mark_edge_freed(lin, i, slot)
                    edges_removed.append((lin, i, slot, o))

        self.cooldown.mark_neuron_freed(layer, slot)
        return {
            "layer": layer,
            "slot": int(slot),
            "neuron_origin": origin,
            "birth_kind": birth_kind,
            "parent_slot": parent,
            "birth_step": bstep,
            "x": x,
            "y": y,
            "edges_removed": edges_removed,
        }

    @torch.no_grad()
    def activate_neuron(
        self,
        layer: str,
        slot: int | None = None,
        *,
        origin: int = NEURON_ORIGIN_GROWN,
        optimizer: Optimizer | None = None,
        coord: Tensor | None = None,
        birth_bias: float = 0.0,
        parent_slot: int | None = None,
        weight_init: str = "kaiming",
        parent_noise: float = 0.1,
        copy_outgoing: bool = True,
        parent_coord_jitter: float = 0.05,
        gap_weight_row: Tensor | None = None,
    ) -> dict:
        """Activate a free slot; reinit, parent-seed, or gap-imprint weights.

        ``birth_bias``: temporary positive pre-activation scaffold (decayed by
        :meth:`update_birth_biases`); not a learned Parameter.

        ``weight_init``:
          - ``"kaiming"``: independent Kaiming-style row/col init (default).
          - ``"parent_perturbed"``: **clone** the growth-trigger unit's incident
            weights and edge masks with noise (requires ``parent_slot``). This is
            *not* gap content — it copies whatever the trigger unit already
            computes.
          - ``"gap_imprinted"``: incoming row set from ``gap_weight_row`` (caller
            builds residual pre-act direction on CE-excess samples via
            :func:`neuronca.plasticity.rules.gap_imprint_direction`). Outgoing is
            Kaiming; edges are *not* copied from the trigger. ``parent_slot`` only
            places the coord near the trigger locus.
        """
        active = self.active_mask(layer)
        if slot is None:
            free = (~active).nonzero(as_tuple=False).view(-1)
            if free.numel() == 0:
                raise RuntimeError(f"no free slots in {layer}")
            # Prefer lowest free index that is cooldown-eligible.
            slot = None
            for cand in free.tolist():
                if self.cooldown.neuron_eligible(layer, cand):
                    slot = int(cand)
                    break
            if slot is None:
                raise RuntimeError(
                    f"no cooldown-eligible free slots in {layer} "
                    f"(blocked={self.cooldown.same_step_blocks})"
                )
        else:
            slot = int(slot)
            if bool(active[slot]):
                raise ValueError(f"{layer}[{slot}] already active")
            if not self.cooldown.neuron_eligible(layer, slot):
                raise RuntimeError(
                    f"{layer}[{slot}] on cooldown until next plasticity step"
                )

        mode = str(weight_init).strip().lower()
        use_parent = mode in ("parent_perturbed", "parent", "copy_parent")
        use_gap = mode in ("gap_imprinted", "gap_imprint", "gap")
        parent: int | None = None
        locus = int(parent_slot) if parent_slot is not None else None
        if use_parent:
            if parent_slot is None:
                raise ValueError(
                    "weight_init='parent_perturbed' requires parent_slot "
                    "(growth-trigger index)"
                )
            parent = int(parent_slot)
            if parent == slot:
                raise ValueError("parent_slot must differ from newborn slot")
            if not bool(active[parent]):
                raise ValueError(
                    f"parent {layer}[{parent}] is not active; cannot seed from it"
                )
        if use_gap:
            if gap_weight_row is None:
                raise ValueError(
                    "weight_init='gap_imprinted' requires gap_weight_row "
                    "(CE-excess residual pre-act direction, already scaled)"
                )
        # Place near trigger locus for structured locality (parent *or* gap).
        if coord is None and locus is not None and parent_coord_jitter >= 0:
            if 0 <= locus < active.numel() and bool(active[locus]):
                base = self.coords(layer)[locus].float()
                if torch.isfinite(base).all():
                    jitter = parent_coord_jitter
                    noise = torch.randn(2, device=base.device, dtype=base.dtype) * jitter
                    coord = (base + noise).clamp(0.0, 1.0)

        if use_gap:
            resolved_init = "gap_imprinted"
        elif use_parent:
            resolved_init = "parent_perturbed"
        else:
            resolved_init = "kaiming"
        kind_code = int(
            BIRTH_KIND_FROM_WEIGHT_INIT.get(resolved_init, NEURON_BIRTH_NONE)
        )
        locus_for_meta = parent if use_parent else locus

        # Birth at current plasticity step (after begin_plasticity_step in orch).
        self._birth_neuron(
            layer,
            slot,
            origin,
            coord=coord,
            birth_step=int(self.cooldown.plasticity_step),
            birth_kind=kind_code,
            parent_slot=locus_for_meta,
        )
        b0 = float(birth_bias)
        self.birth_bias0(layer)[slot] = b0
        self.birth_bias(layer)[slot] = b0

        gap_dir_norm: float | None = None
        # Reinit, parent-seed, or gap-imprint incident weight rows/cols.
        for lin, (pre, post) in _LINEAR_ENDPOINTS.items():
            mod = self.linear_module(lin)
            if post == layer:
                if use_parent and parent is not None:
                    self._copy_weight_row_from_parent(
                        mod, slot, parent, noise=float(parent_noise)
                    )
                    self._copy_edge_row_from_parent(lin, slot, parent)
                elif use_gap:
                    assert gap_weight_row is not None
                    scaled = gap_weight_row.detach().float().reshape(-1)
                    if float(parent_noise) > 0:
                        scaled = self._perturb_like(scaled, float(parent_noise))
                    if scaled.numel() != mod.in_features:
                        raise ValueError(
                            f"gap imprint fan_in {scaled.numel()} != "
                            f"layer {mod.in_features}"
                        )
                    mod.weight[slot].copy_(
                        scaled.to(device=mod.weight.device, dtype=mod.weight.dtype)
                    )
                    if mod.bias is not None:
                        mod.bias[slot] = 0.0
                    gap_dir_norm = float(scaled.norm().item())
                    # Do NOT copy edge fan from trigger — gap content ≠ clone topology.
                else:
                    self._reinit_weight_row(mod, slot)
                if optimizer is not None:
                    reset_linear_neuron_optimizer_state(
                        optimizer, mod.weight, mod.bias, out_index=slot
                    )
            if pre == layer:
                if use_parent and parent is not None and copy_outgoing:
                    self._copy_weight_col_from_parent(
                        mod, slot, parent, noise=float(parent_noise)
                    )
                    self._copy_edge_col_from_parent(lin, slot, parent)
                else:
                    # gap_imprinted and kaiming: fresh outgoing, no parent col copy.
                    self._reinit_weight_col(mod, slot)
                if optimizer is not None:
                    reset_optimizer_state_for_indices(
                        optimizer, mod.weight, col_indices=[slot]
                    )

        return {
            "layer": layer,
            "slot": slot,
            "neuron_origin": origin,
            "birth_bias": b0,
            "parent_slot": locus_for_meta,
            "weight_init": resolved_init,
            "birth_kind": BIRTH_KIND_NAMES.get(kind_code, "none"),
            "gap_imprint_norm": gap_dir_norm,
        }

    @staticmethod
    def _reinit_weight_row(mod: nn.Linear, row: int) -> None:
        with torch.no_grad():
            # Kaiming-like per-row: sample fan_in from in_features.
            fan_in = mod.in_features
            bound = (6.0 / fan_in) ** 0.5  # approx kaiming_uniform gain for relu
            mod.weight[row].uniform_(-bound, bound)
            if mod.bias is not None:
                mod.bias[row] = 0.0

    @staticmethod
    def _reinit_weight_col(mod: nn.Linear, col: int) -> None:
        with torch.no_grad():
            fan_in = mod.in_features
            bound = (6.0 / max(fan_in, 1)) ** 0.5
            mod.weight[:, col].uniform_(-bound, bound)

    @staticmethod
    def _perturb_like(src: Tensor, noise: float) -> Tensor:
        """Perturbed copy: multiplicative + additive noise scaled by src std."""
        if noise <= 0:
            return src.clone()
        std = float(src.float().std().clamp(min=1e-6).item())
        eps_mul = torch.randn_like(src) * float(noise)
        eps_add = torch.randn_like(src) * (float(noise) * std)
        return src * (1.0 + eps_mul) + eps_add

    def _copy_weight_row_from_parent(
        self, mod: nn.Linear, row: int, parent: int, *, noise: float
    ) -> None:
        with torch.no_grad():
            mod.weight[row].copy_(self._perturb_like(mod.weight[parent].detach(), noise))
            if mod.bias is not None:
                # Bias: small relative jitter only (scalar).
                b = float(mod.bias[parent].item())
                if noise > 0:
                    b = b * (1.0 + float(noise) * torch.randn(()).item())
                mod.bias[row] = b

    def _copy_weight_col_from_parent(
        self, mod: nn.Linear, col: int, parent: int, *, noise: float
    ) -> None:
        with torch.no_grad():
            mod.weight[:, col].copy_(
                self._perturb_like(mod.weight[:, parent].detach(), noise)
            )

    def _copy_edge_row_from_parent(self, lin: str, row: int, parent: int) -> None:
        """Copy postsynaptic edge row (incoming fan) from parent → newborn."""
        em = self.edge_mask(lin)
        eo = self.edge_origin(lin)
        em[row].copy_(em[parent])
        # Tag copied structure as incentive-grown for provenance.
        present = em[row]
        eo[row] = torch.where(
            present,
            torch.full_like(eo[row], EDGE_ORIGIN_INCENTIVE),
            torch.zeros_like(eo[row]),
        )

    def _copy_edge_col_from_parent(self, lin: str, col: int, parent: int) -> None:
        """Copy presynaptic edge column (outgoing fan) from parent → newborn."""
        em = self.edge_mask(lin)
        eo = self.edge_origin(lin)
        em[:, col].copy_(em[:, parent])
        present = em[:, col]
        eo[:, col] = torch.where(
            present,
            torch.full_like(eo[:, col], EDGE_ORIGIN_INCENTIVE),
            torch.zeros_like(eo[:, col]),
        )

    @torch.no_grad()
    def scale_newborn_critical_grads(
        self,
        lr_mult: float,
        grace_steps: int,
    ) -> None:
        """Scale grads on newborn incident weights (critical-period analog).

        While ``plasticity_step - birth_step < grace_steps``, multiply the
        gradient on that slot's weight row/col (and bias) by ``lr_mult``.
        No-op when ``lr_mult == 1`` or ``grace_steps <= 0``. Distinct from
        extending grace (same rule, more time).
        """
        if lr_mult == 1.0 or grace_steps <= 0:
            return
        mult = float(lr_mult)
        step = int(self.cooldown.plasticity_step)
        for h in HIDDEN_LAYERS:
            birth = self.birth_step(h)
            origin = self.neuron_origin(h)
            active = self.active_mask(h)
            # Grown units still inside critical window.
            age = step - birth
            in_cp = (
                active
                & (origin == NEURON_ORIGIN_GROWN)
                & (birth >= 0)
                & (age >= 0)
                & (age < int(grace_steps))
            )
            if not bool(in_cp.any()):
                continue
            slots = in_cp.nonzero(as_tuple=False).view(-1).tolist()
            for lin, (pre, post) in _LINEAR_ENDPOINTS.items():
                mod = self.linear_module(lin)
                w = mod.weight
                if w.grad is None:
                    continue
                if post == h:
                    for s in slots:
                        w.grad[s].mul_(mult)
                        if mod.bias is not None and mod.bias.grad is not None:
                            mod.bias.grad[s].mul_(mult)
                if pre == h:
                    for s in slots:
                        w.grad[:, s].mul_(mult)

    @torch.no_grad()
    def remove_edge(self, layer: str, out_idx: int, in_idx: int) -> dict:
        """Prune a single edge; record origin and cooldown."""
        em = self.edge_mask(layer)
        eo = self.edge_origin(layer)
        if not bool(em[out_idx, in_idx]):
            raise ValueError(f"edge {layer}[{out_idx},{in_idx}] not present")
        origin = int(eo[out_idx, in_idx].item())
        em[out_idx, in_idx] = False
        eo[out_idx, in_idx] = EDGE_ORIGIN_NONE
        self.cooldown.mark_edge_freed(layer, out_idx, in_idx)
        return {
            "layer": layer,
            "out": int(out_idx),
            "in": int(in_idx),
            "origin": origin,
        }

    @torch.no_grad()
    def add_edge(
        self,
        layer: str,
        out_idx: int,
        in_idx: int,
        *,
        origin: int = EDGE_ORIGIN_INCENTIVE,
        optimizer: Optimizer | None = None,
        reinit_weight: bool = True,
    ) -> dict:
        """Form an edge if eligible (not present, not on cooldown)."""
        em = self.edge_mask(layer)
        eo = self.edge_origin(layer)
        if bool(em[out_idx, in_idx]):
            raise ValueError(f"edge {layer}[{out_idx},{in_idx}] already present")
        if not self.cooldown.edge_eligible(layer, out_idx, in_idx):
            raise RuntimeError(
                f"edge {layer}[{out_idx},{in_idx}] on cooldown until next step"
            )
        pre, post = _LINEAR_ENDPOINTS[layer]
        if not bool(self.population_active(post)[out_idx]):
            raise ValueError(f"post unit {post}[{out_idx}] inactive")
        if not bool(self.population_active(pre)[in_idx]):
            raise ValueError(f"pre unit {pre}[{in_idx}] inactive")

        em[out_idx, in_idx] = True
        eo[out_idx, in_idx] = origin
        mod = self.linear_module(layer)
        if reinit_weight:
            fan_in = mod.in_features
            bound = (6.0 / max(fan_in, 1)) ** 0.5
            mod.weight[out_idx, in_idx] = (
                torch.empty((), device=mod.weight.device).uniform_(-bound, bound)
            )
            if optimizer is not None:
                reset_linear_edge_optimizer_state(
                    optimizer, mod.weight, out_idx, in_idx
                )
        return {
            "layer": layer,
            "out": int(out_idx),
            "in": int(in_idx),
            "origin": origin,
        }

    def free_slot_count(self, layer: str) -> int:
        return int((~self.active_mask(layer)).sum())

    def at_capacity(self, layer: str) -> bool:
        return self.free_slot_count(layer) == 0
