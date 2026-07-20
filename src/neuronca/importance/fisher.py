"""GT-3: diagonal Fisher information (optional).

    F_ij ≈ E_probe[ (∂L / ∂W_ji)^2 ]

Enabled when ``gt.fisher: true``. Same dense layout as GT-1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from neuronca.importance.base import (
    ImportanceResult,
    cpu_float,
    iter_linear_layers,
    resolve_device,
    unpack_batch,
)


class FisherImportance:
    """Diagonal empirical Fisher on probe (optional GT). Name: ``fisher``."""

    name: str = "fisher"

    def compute(
        self,
        model: nn.Module,
        probe_loader: DataLoader,
        criterion: nn.Module,
        *,
        edge_sample_mask: dict[str, Tensor] | None = None,
    ) -> ImportanceResult:
        """Accumulate mean squared weight gradients over the probe set.

        ``edge_sample_mask`` is ignored (dense). Does not call local-signal updates.
        """
        del edge_sample_mask
        device = resolve_device(model)
        was_training = model.training
        model.eval()
        model.to(device)

        layers = iter_linear_layers(model)
        if not layers:
            raise ValueError("model has no nn.Linear layers to score")

        sum_sq: dict[str, Tensor] = {
            name: torch.zeros_like(mod.weight, device=device) for name, mod in layers
        }
        total_n = 0

        for batch in probe_loader:
            x, y = unpack_batch(batch)
            x = x.to(device)
            y = y.to(device)
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
                sum_sq[name] = sum_sq[name] + g.detach().square() * bs
            total_n += bs

        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()

        if total_n == 0:
            raise ValueError("probe_loader yielded no examples")

        edge: dict[str, Tensor] = {}
        neuron: dict[str, Tensor] = {}
        for name, _mod in layers:
            f_edge = sum_sq[name] / float(total_n)
            edge[name] = cpu_float(f_edge)
            neuron[name] = cpu_float(f_edge.sum(dim=1))

        return ImportanceResult(
            neuron=neuron,
            edge=edge,
            edge_index=None,
            extras={},
        )


def compute_fisher(
    model: nn.Module,
    probe_loader: DataLoader,
    criterion: nn.Module,
) -> ImportanceResult:
    """Functional wrapper around :class:`FisherImportance`."""
    return FisherImportance().compute(model, probe_loader, criterion)
