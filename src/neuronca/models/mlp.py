"""Fixed-topology MLP reasoning network with standardized activation hooks."""

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MLPReasoningNet(nn.Module):
    """MLP: input → 256 → 128 → 64 → num_classes with named Linear layers.

    Layers are named exactly ``fc1``, ``fc2``, ``fc3``, ``fc_out``.
    Hidden layers use ReLU; ``fc_out`` produces raw logits.

    Activation keys (when collected):
        act_in     — flattened input
        act_fc1    — post-ReLU after fc1
        act_fc2    — post-ReLU after fc2
        act_fc3    — post-ReLU after fc3
        act_fc_out — logits (no nonlinearity)

    Every forward populates ``self.last_activations`` with **detached clones**
    (no autograd graph; values stable against later in-place edits of the input).
    When ``return_activations=True``, the returned dict holds live graph-bearing
    tensors so callers that need them can still use them for diagnostics;
    training should always backprop through the returned logits tensor alone.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden: tuple[int, ...] = (256, 128, 64),
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        if len(hidden) != 3:
            raise ValueError(
                f"hidden must have exactly 3 sizes (fc1, fc2, fc3); got {hidden!r}"
            )
        h1, h2, h3 = hidden
        self.input_dim = input_dim
        self.hidden = hidden
        self.num_classes = num_classes

        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc_out = nn.Linear(h3, num_classes)

        self._init_weights()
        self.last_activations: dict[str, Tensor] | None = None

    def _init_weights(self) -> None:
        for name, module in self.named_linear_layers():
            # Kaiming uniform for ReLU hidden; use same for logits layer for simplicity.
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def named_linear_layers(self) -> list[tuple[str, nn.Linear]]:
        """Ordered Linear modules: fc1, fc2, fc3, fc_out."""
        return [
            ("fc1", self.fc1),
            ("fc2", self.fc2),
            ("fc3", self.fc3),
            ("fc_out", self.fc_out),
        ]

    def layer_shapes(self) -> dict[str, tuple[int, int]]:
        """Map layer name → (n_out, n_in) matching ``Linear.weight`` layout.

        Used by the signals API for score allocation; no weight values.
        """
        return {
            name: (mod.out_features, mod.in_features)
            for name, mod in self.named_linear_layers()
        }

    def forward(
        self,
        x: Tensor,
        return_activations: bool = False,
    ) -> Union[Tensor, tuple[Tensor, dict[str, Tensor]]]:
        """Forward pass.

        Args:
            x: Input of shape ``(B, 784)`` or ``(B, 1, 28, 28)`` / ``(B, 28, 28)``.
            return_activations: If True, also return a live activation dict
                (graph-bearing tensors). The module cache ``last_activations``
                always stores detached clones regardless of this flag.

        Returns:
            Logits ``(B, num_classes)``, or ``(logits, activations)`` when
            ``return_activations`` is True. Always stores detached activation
            snapshots on ``self.last_activations``.
        """
        # Flatten image inputs to (B, input_dim).
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"expected last dim {self.input_dim}, got shape {tuple(x.shape)}"
            )

        act_in = x
        h1 = F.relu(self.fc1(act_in))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        logits = self.fc_out(h3)

        # Live tensors share the compute graph with logits (for optional return).
        activations: dict[str, Tensor] = {
            "act_in": act_in,
            "act_fc1": h1,
            "act_fc2": h2,
            "act_fc3": h3,
            "act_fc_out": logits,
        }
        # Detached clones: no graph retention after backward; act_in snapshot is
        # stable if the caller's batch buffer is later mutated in-place.
        self.last_activations = {k: v.detach().clone() for k, v in activations.items()}

        if return_activations:
            return logits, activations
        return logits
