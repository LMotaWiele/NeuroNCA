"""Ground-truth importance protocol and shared result types.

GT modules may use full gradient / ablation access. They must **never** call
``LocalSignal.update`` or mutate local-signal traces (snapshot isolation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Union, runtime_checkable

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

# Live DataLoader or snapshot-scoped ProbeCache (string form avoids import cycle).
ProbeSource = Union[DataLoader, "ProbeCache"]


@dataclass
class ImportanceResult:
    """Per-layer neuron/edge importance tensors from a GT measure.

    Attributes:
        neuron: ``fc*`` → ``[n_out]`` scores (post units).
        edge: ``fc*`` → ``[n_out, n_in]`` scores matching ``Linear.weight`` layout.
        edge_index: Optional per-layer index/mask of scored edges (sparse GT-2);
            ``None`` for dense measures (GT-1, Fisher).
        extras: Named secondary columns, e.g. ``grad_movement`` (layer→tensor) or
            nested ``ablation_signed`` ``{"neuron": {...}, "edge": {...}}``.
    """

    neuron: dict[str, Tensor]
    edge: dict[str, Tensor]
    edge_index: dict[str, Tensor] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class GroundTruthImportance(Protocol):
    """GT importance measure computed on a frozen probe loader."""

    name: str

    def compute(
        self,
        model: nn.Module,
        probe_loader: ProbeSource,
        criterion: nn.Module,
        *,
        edge_sample_mask: dict[str, Tensor] | None = None,
    ) -> ImportanceResult: ...


def iter_linear_layers(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    """Ordered ``(name, Linear)`` pairs; prefer ``named_linear_layers`` when present."""
    named = getattr(model, "named_linear_layers", None)
    if callable(named):
        layers = list(named())
        if layers:
            return layers
    return [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
    ]


def unpack_batch(batch: Any) -> tuple[Tensor, Tensor]:
    """Unpack ``(x, y)`` or ``(x, y, ...)`` from a DataLoader batch."""
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError(
            f"expected batch as (x, y) or (x, y, ...); got {type(batch)!r}"
        )
    return batch[0], batch[1]


def resolve_device(model: nn.Module) -> torch.device:
    """Device of the first model parameter (fallback CPU)."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def mean_probe_loss(
    model: nn.Module,
    probe_loader: ProbeSource,
    criterion: nn.Module,
    *,
    device: torch.device | None = None,
) -> float:
    """Mean criterion over the full probe set (eval mode, no grad).

    ``probe_loader`` may be a :class:`~torch.utils.data.DataLoader` or a
    snapshot-scoped :class:`~neuronca.importance.probe_cache.ProbeCache`.
    """
    # Local import avoids circular dependency at module load.
    from neuronca.importance.probe_cache import iter_probe_batches

    if device is None:
        device = resolve_device(model)
    was_training = model.training
    model.eval()
    total = 0.0
    n = 0
    # Loader batches are moved to ``device``; ProbeCache batches already on-device.
    for x, y in iter_probe_batches(probe_loader, device=device):
        # Flatten integer class-label tensors only (CE); keep float (B, C) for MSE.
        if y.ndim > 1 and not y.is_floating_point():
            y = y.view(-1)
        logits = model(x)
        loss = criterion(logits, y)
        bs = int(y.size(0))
        total += float(loss.item()) * bs
        n += bs
    if was_training:
        model.train()
    if n == 0:
        return float("nan")
    return total / n


def cpu_float(t: Tensor) -> Tensor:
    """Detach → CPU float32 clone for stable artifact / correlation use."""
    return t.detach().float().cpu().clone()
