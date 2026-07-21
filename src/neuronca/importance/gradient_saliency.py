"""GT-1: gradient-based saliency (``grad_abs``).

On the current-task probe loader:

    I^{grad}_{ij} = E_probe[ |∂L / ∂W_ji| ]

Neuron-level: sum of incoming abs-grad mass. Also log ``grad_movement`` =
|∂L/∂W| · |W| as a secondary column.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from neuronca.importance.base import (
    ImportanceResult,
    ProbeSource,
    cpu_float,
    iter_linear_layers,
    resolve_device,
)
from neuronca.importance.probe_cache import iter_probe_batches


class GradientSaliency:
    """Dense gradient saliency (primary GT). Name: ``grad_abs``."""

    name: str = "grad_abs"

    def __init__(self, *, grad_batch_size: int | None = None) -> None:
        """Args:
            grad_batch_size: Optional hint only (probe loader already batched).
                Kept for config parity with ``gt.grad_batch_size``.
        """
        self.grad_batch_size = grad_batch_size

    def compute(
        self,
        model: nn.Module,
        probe_loader: ProbeSource,
        criterion: nn.Module,
        *,
        edge_sample_mask: dict[str, Tensor] | None = None,
    ) -> ImportanceResult:
        """Accumulate mean |weight grad| over the probe set.

        ``probe_loader`` may be a DataLoader or a snapshot-scoped
        :class:`~neuronca.importance.probe_cache.ProbeCache`.

        ``edge_sample_mask`` is accepted for protocol compatibility and ignored
        (GT-1 is always dense). Does not call any local-signal update path.
        """
        del edge_sample_mask  # dense measure
        device = resolve_device(model)
        was_training = model.training
        model.eval()
        model.to(device)

        layers = iter_linear_layers(model)
        if not layers:
            raise ValueError("model has no nn.Linear layers to score")

        # Accumulators on the parameter device.
        sum_abs_grad: dict[str, Tensor] = {}
        weight_ref: dict[str, Tensor] = {}
        for name, mod in layers:
            w = mod.weight
            sum_abs_grad[name] = torch.zeros_like(w, device=device)
            weight_ref[name] = w.detach()

        total_n = 0
        for x, y in iter_probe_batches(probe_loader, device=device):
            # Flatten integer class labels only; keep float (B, C) for MSE-style criteria.
            if y.ndim > 1 and not y.is_floating_point():
                y = y.view(-1)
            bs = int(x.size(0))
            if bs == 0:
                continue

            model.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            for name, mod in layers:
                g = mod.weight.grad
                if g is None:
                    continue
                # Weight mean-loss grads by batch size so last partial batch is fair.
                sum_abs_grad[name] = sum_abs_grad[name] + g.detach().abs() * bs
            total_n += bs

        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()

        if total_n == 0:
            raise ValueError("probe_loader yielded no examples")

        edge: dict[str, Tensor] = {}
        neuron: dict[str, Tensor] = {}
        grad_movement: dict[str, Tensor] = {}

        for name, mod in layers:
            # I_ij = E[|∂L/∂W|]; L is mean-reduced per mini-batch.
            i_edge = sum_abs_grad[name] / float(total_n)
            # Neuron: sum incoming abs-grad mass (dim=1 = n_in).
            i_neuron = i_edge.sum(dim=1)
            w_abs = weight_ref[name].abs()
            movement = i_edge * w_abs

            edge[name] = cpu_float(i_edge)
            neuron[name] = cpu_float(i_neuron)
            grad_movement[name] = cpu_float(movement)

        return ImportanceResult(
            neuron=neuron,
            edge=edge,
            edge_index=None,
            extras={"grad_movement": grad_movement},
        )


def compute_grad_abs(
    model: nn.Module,
    probe_loader: ProbeSource,
    criterion: nn.Module,
    *,
    grad_batch_size: int | None = None,
) -> ImportanceResult:
    """Functional wrapper around :class:`GradientSaliency`."""
    return GradientSaliency(grad_batch_size=grad_batch_size).compute(
        model, probe_loader, criterion
    )
