"""Candidate 1 — Activation magnitude / frequency baseline.

Neuron j:  s_j  <- λ s_j  + (1-λ) mean_b |a_j|
Edge i→j:  s_ij <- λ s_ij + (1-λ) mean_b |a_i| |a_j|

No dependence on broadcast. Weight values are never read.
"""

from __future__ import annotations

import torch
from torch import Tensor

from neuronca.broadcast.error import BroadcastPacket
from neuronca.signals.base import (
    DEFAULT_DECAY,
    _BaseSignal,
    ema_update,
    get_pre_post,
    validate_layer_shapes,
)


class ActivationMagnitude(_BaseSignal):
    """EMA of mean absolute activations (neuron) and |a_i||a_j| (edge)."""

    name = "activation"

    def __init__(self, decay: float = DEFAULT_DECAY) -> None:
        super().__init__(decay=decay)

    @torch.no_grad()
    def update(
        self,
        activations: dict[str, Tensor],
        layer_shapes: dict[str, tuple[int, int]],
        broadcast: BroadcastPacket,
        step: int,
    ) -> None:
        del broadcast  # unused; baseline ignores error broadcast
        if not layer_shapes:
            raise ValueError("layer_shapes must be non-empty")
        validate_layer_shapes(layer_shapes, activations)

        # Device/dtype from any activation
        sample = next(iter(activations.values()))
        self._ensure_buffers(layer_shapes, device=sample.device, dtype=torch.float32)
        decay = self.decay

        for layer, (n_out, n_in) in layer_shapes.items():
            a_pre, a_post = get_pre_post(activations, layer)
            B = a_pre.shape[0]
            a_pre = a_pre.detach().to(dtype=torch.float32)
            a_post = a_post.detach().to(dtype=torch.float32)

            # Neuron: mean_b |a_j|
            neuron_stat = a_post.abs().mean(dim=0)  # [n_out]
            assert neuron_stat.shape == (n_out,)
            ema_update(self._neuron[layer], neuron_stat, decay)

            # Edge: mean_b |a_i| |a_j| → [n_out, n_in]
            abs_pre = a_pre.abs()  # [B, n_in]
            abs_post = a_post.abs()  # [B, n_out]
            edge_stat = (abs_post.transpose(0, 1) @ abs_pre) / float(B)
            assert edge_stat.shape == (n_out, n_in)
            ema_update(self._edge[layer], edge_stat, decay)

        self._step = int(step)
