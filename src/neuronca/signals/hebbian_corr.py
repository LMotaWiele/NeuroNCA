"""Candidate 2 — Hebbian correlation with residual energy.

EMA of factors and product, then centered covariance:

    m_a  <- λ m_a  + (1-λ) mean_b a
    m_e  <- λ m_e  + (1-λ) mean_b e_s
    m_ae <- λ m_ae + (1-λ) mean_b a e_s
    c = m_ae - m_a m_e
    s = |c|

Neurons: a^{(b)} = a_j^{(b)}.
Edges:   a^{(b)} = a_i^{(b)} a_j^{(b)} (per-sample product).

Uses per-sample residual energy only (never batch-scalar e).
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


class HebbianResidualEnergyCorr(_BaseSignal):
    """Centered EMA covariance of activity (or co-activation) with residual energy."""

    name = "hebbian_corr"

    def __init__(self, decay: float = DEFAULT_DECAY) -> None:
        super().__init__(decay=decay)
        self._m_e: Tensor | None = None
        self._m_a_neuron: dict[str, Tensor] = {}
        self._m_ae_neuron: dict[str, Tensor] = {}
        self._m_a_edge: dict[str, Tensor] = {}
        self._m_ae_edge: dict[str, Tensor] = {}

    def _on_reset(self) -> None:
        self._m_e = None
        self._m_a_neuron.clear()
        self._m_ae_neuron.clear()
        self._m_a_edge.clear()
        self._m_ae_edge.clear()

    def _on_allocate(
        self,
        layer_shapes: Mapping[str, tuple[int, int]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self._m_e = torch.zeros((), device=device, dtype=dtype)
        for layer, (n_out, n_in) in layer_shapes.items():
            self._m_a_neuron[layer] = torch.zeros(n_out, device=device, dtype=dtype)
            self._m_ae_neuron[layer] = torch.zeros(n_out, device=device, dtype=dtype)
            self._m_a_edge[layer] = torch.zeros(n_out, n_in, device=device, dtype=dtype)
            self._m_ae_edge[layer] = torch.zeros(n_out, n_in, device=device, dtype=dtype)

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
            context="hebbian_corr",
        )
        e_s = e_s.detach().to(device=sample.device, dtype=torch.float32)

        self._ensure_buffers(layer_shapes, device=sample.device, dtype=torch.float32)
        assert self._m_e is not None
        decay = self.decay

        # Shared residual-energy EMA (scalar)
        bar_e = e_s.mean()
        ema_update(self._m_e, bar_e, decay)
        m_e = self._m_e

        for layer, (n_out, n_in) in layer_shapes.items():
            a_pre, a_post = get_pre_post(activations, layer)
            a_pre = a_pre.detach().to(dtype=torch.float32)
            a_post = a_post.detach().to(dtype=torch.float32)

            # --- Neurons: a = a_j ---
            bar_a_n = a_post.mean(dim=0)  # [n_out]
            bar_ae_n = (a_post * e_s[:, None]).mean(dim=0)  # [n_out]
            ema_update(self._m_a_neuron[layer], bar_a_n, decay)
            ema_update(self._m_ae_neuron[layer], bar_ae_n, decay)
            c_n = self._m_ae_neuron[layer] - self._m_a_neuron[layer] * m_e
            self._neuron[layer].copy_(c_n.abs())

            # --- Edges: a = a_i a_j (per-sample product) ---
            # mean_b a_i a_j = (a_post^T @ a_pre) / B
            bar_a_e = (a_post.transpose(0, 1) @ a_pre) / float(B)  # [n_out, n_in]
            # mean_b a_i a_j e = ((a_post * e)^T @ a_pre) / B
            weighted_post = a_post * e_s[:, None]
            bar_ae_e = (weighted_post.transpose(0, 1) @ a_pre) / float(B)
            ema_update(self._m_a_edge[layer], bar_a_e, decay)
            ema_update(self._m_ae_edge[layer], bar_ae_e, decay)
            c_e = self._m_ae_edge[layer] - self._m_a_edge[layer] * m_e
            self._edge[layer].copy_(c_e.abs())

            assert self._neuron[layer].shape == (n_out,)
            assert self._edge[layer].shape == (n_out, n_in)

        self._step = int(step)
