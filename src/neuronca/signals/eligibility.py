"""Candidate 3 — Three-factor eligibility with scalar residual energy.

Name: ``eligibility_residual_energy`` (not GEVB).

Connection (i→j):
    δ_ij = mean_b a_i a_j e_s^{(b)}
    T_ij <- λ T_ij + (1-λ) δ_ij
    s_ij = |T_ij|

Neuron j (mean abs incoming):
    s_j = (1 / n_in) sum_i |T_ij|

Must use per-sample e_s; batch-scalar e with per-sample a products raises ConfigError.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor

from neuronca.broadcast.error import BroadcastPacket
from neuronca.signals.base import (
    DEFAULT_DECAY,
    _BaseSignal,
    ema_update,
    get_pre_post,
    require_per_sample_modulator,
    validate_layer_shapes,
)


class ThreeFactorScalarEligibility(_BaseSignal):
    """Three-factor residual-energy eligibility (scalar neuromodulator path)."""

    name = "eligibility_residual_energy"

    def __init__(self, decay: float = DEFAULT_DECAY) -> None:
        super().__init__(decay=decay)
        # T traces (signed); exposed scores are |T|
        self._T_edge: dict[str, Tensor] = {}
        self._neuron_out_mean: dict[str, Tensor] = {}  # diagnostic: mean abs outgoing

    def _on_reset(self) -> None:
        self._T_edge.clear()
        self._neuron_out_mean.clear()

    def _on_allocate(
        self,
        layer_shapes: Mapping[str, tuple[int, int]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        for layer, (n_out, n_in) in layer_shapes.items():
            self._T_edge[layer] = torch.zeros(n_out, n_in, device=device, dtype=dtype)
            self._neuron_out_mean[layer] = torch.zeros(n_out, device=device, dtype=dtype)

    @torch.no_grad()
    def update(
        self,
        activations: dict[str, Tensor],
        layer_shapes: dict[str, tuple[int, int]],
        broadcast: BroadcastPacket,
        step: int,
    ) -> None:
        if not layer_shapes:
            raise ValueError("layer_shapes must be non-empty")
        validate_layer_shapes(layer_shapes, activations)

        sample = next(iter(activations.values()))
        B = int(sample.shape[0])
        e_s = require_per_sample_modulator(
            broadcast.eligibility_energy_modulator(),
            B,
            context="eligibility_residual_energy",
        )
        e_s = e_s.detach().to(device=sample.device, dtype=torch.float32)

        self._ensure_buffers(layer_shapes, device=sample.device, dtype=torch.float32)
        decay = self.decay

        for layer, (n_out, n_in) in layer_shapes.items():
            a_pre, a_post = get_pre_post(activations, layer)
            a_pre = a_pre.detach().to(dtype=torch.float32)
            a_post = a_post.detach().to(dtype=torch.float32)

            # δ_ij = mean_b a_i a_j e_s^{(b)}
            weighted_post = a_post * e_s[:, None]  # [B, n_out]
            delta = (weighted_post.transpose(0, 1) @ a_pre) / float(B)  # [n_out, n_in]
            assert delta.shape == (n_out, n_in)

            ema_update(self._T_edge[layer], delta, decay)
            T = self._T_edge[layer]
            abs_T = T.abs()
            self._edge[layer].copy_(abs_T)

            # Neuron: mean abs incoming over presynaptic index i
            # abs_T is [n_out, n_in]; mean over dim=1 (incoming)
            self._neuron[layer].copy_(abs_T.mean(dim=1))

            # Diagnostic: mean abs outgoing per post unit is same as incoming mean
            # for row-wise mean; also log mean abs over outgoing columns if needed.
            # Design: "Also log mean abs outgoing for analysis."
            # Outgoing for pre unit i would be mean_j |T_ij|; store row means already
            # as neuron. Keep column means under extras path via attribute.
            self._neuron_out_mean[layer] = abs_T.mean(dim=0)  # [n_in]

        self._step = int(step)

    def snapshot_scores(self):
        bundle = super().snapshot_scores()
        bundle.extras["neuron_mean_abs_outgoing"] = {
            k: v.detach().to(device="cpu", dtype=torch.float32).clone()
            for k, v in self._neuron_out_mean.items()
        }
        return bundle
