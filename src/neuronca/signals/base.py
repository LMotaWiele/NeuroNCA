"""Local importance signal protocol and shared helpers.

Signals observe activations + broadcast only; they never read weight values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable

import torch
from torch import Tensor

from neuronca.broadcast.error import BroadcastPacket

# Linear name -> (presynaptic activation key, postsynaptic activation key)
LAYER_ACT_KEYS: dict[str, tuple[str, str]] = {
    "fc1": ("act_in", "act_fc1"),
    "fc2": ("act_fc1", "act_fc2"),
    "fc3": ("act_fc2", "act_fc3"),
    "fc_out": ("act_fc3", "act_fc_out"),
}

DEFAULT_DECAY: float = 0.99


class ConfigError(ValueError):
    """Invalid signal / broadcast configuration combination."""


@dataclass
class ScoreBundle:
    """Deep-copied local scores frozen for correlation / GT isolation."""

    name: str
    neuron: dict[str, Tensor]  # fc* -> [n_out] CPU float
    edge: dict[str, Tensor]  # fc* -> [n_out, n_in] CPU float
    step: int | None = None
    extras: dict = field(default_factory=dict)


@runtime_checkable
class LocalSignal(Protocol):
    """Read-only local importance tracker (activations + broadcast only)."""

    name: str

    def reset(self) -> None: ...

    def update(
        self,
        activations: dict[str, Tensor],
        layer_shapes: dict[str, tuple[int, int]],
        broadcast: BroadcastPacket,
        step: int,
    ) -> None: ...

    def neuron_scores(self) -> dict[str, Tensor]:
        """fc* -> [n_out] CPU float."""
        ...

    def edge_scores(self) -> dict[str, Tensor]:
        """fc* -> [n_out, n_in] CPU float."""
        ...

    def snapshot_scores(self) -> ScoreBundle:
        """Deep-copy scores for correlation; used before GT."""
        ...


def pre_post_keys(layer: str) -> tuple[str, str]:
    """Return (pre_key, post_key) for a Linear layer name."""
    if layer not in LAYER_ACT_KEYS:
        raise KeyError(
            f"unknown layer {layer!r}; expected one of {sorted(LAYER_ACT_KEYS)}"
        )
    return LAYER_ACT_KEYS[layer]


def get_pre_post(
    activations: Mapping[str, Tensor],
    layer: str,
) -> tuple[Tensor, Tensor]:
    """Fetch pre/post activations for ``layer``; both shaped ``[B, N]``."""
    pre_key, post_key = pre_post_keys(layer)
    if pre_key not in activations:
        raise KeyError(f"missing activation key {pre_key!r} for layer {layer}")
    if post_key not in activations:
        raise KeyError(f"missing activation key {post_key!r} for layer {layer}")
    a_pre = activations[pre_key]
    a_post = activations[post_key]
    if a_pre.ndim != 2 or a_post.ndim != 2:
        raise ValueError(
            f"activations for {layer} must be [B, N]; "
            f"got pre {tuple(a_pre.shape)} post {tuple(a_post.shape)}"
        )
    if a_pre.shape[0] != a_post.shape[0]:
        raise ValueError(
            f"batch mismatch for {layer}: pre B={a_pre.shape[0]} post B={a_post.shape[0]}"
        )
    return a_pre, a_post


def validate_layer_shapes(
    layer_shapes: Mapping[str, tuple[int, int]],
    activations: Mapping[str, Tensor],
) -> None:
    """Check that activation widths match (n_out, n_in) shape metadata."""
    for layer, (n_out, n_in) in layer_shapes.items():
        a_pre, a_post = get_pre_post(activations, layer)
        if a_pre.shape[1] != n_in:
            raise ValueError(
                f"{layer}: n_in={n_in} but pre act width={a_pre.shape[1]}"
            )
        if a_post.shape[1] != n_out:
            raise ValueError(
                f"{layer}: n_out={n_out} but post act width={a_post.shape[1]}"
            )


def require_per_sample_modulator(e_s: Tensor, batch_size: int, *, context: str) -> Tensor:
    """Ensure residual-energy modulator is per-sample ``[B]``, not a batch scalar.

    Pairing a batch-scalar ``e_s`` with per-sample co-activation products collapses
    three-factor structure to ``e_s * mean(a_i a_j)``. Raise :class:`ConfigError`.
    """
    if not torch.is_tensor(e_s):
        raise ConfigError(
            f"{context}: eligibility modulator must be a tensor [B], got {type(e_s)}"
        )
    if e_s.ndim == 0:
        raise ConfigError(
            f"{context}: batch-scalar e_s used with per-sample a products is forbidden "
            f"(would reduce to e_s * mean(a_i a_j)); use per-sample e_s of shape [{batch_size}]"
        )
    if e_s.ndim != 1:
        raise ConfigError(
            f"{context}: e_s must be shape [B], got {tuple(e_s.shape)}"
        )
    if e_s.shape[0] != batch_size:
        # Single-element tensor used as a shared scalar for B>1 is also invalid.
        if e_s.numel() == 1 and batch_size > 1:
            raise ConfigError(
                f"{context}: batch-scalar e_s (numel=1) used with per-sample a products "
                f"for B={batch_size} is forbidden; use per-sample e_s of shape [{batch_size}]"
            )
        raise ConfigError(
            f"{context}: e_s batch {e_s.shape[0]} != activation batch {batch_size}"
        )
    return e_s


def ema_update(trace: Tensor, value: Tensor, decay: float) -> Tensor:
    """In-place EMA: ``trace <- decay * trace + (1 - decay) * value``."""
    # ``trace.mul_(decay).add_(value, alpha=1-decay)`` avoids extra temps.
    return trace.mul_(decay).add_(value, alpha=1.0 - decay)


def scores_to_cpu(scores: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Detach clone scores to CPU float32 for export."""
    return {k: v.detach().to(device="cpu", dtype=torch.float32).clone() for k, v in scores.items()}


def snapshot_dict(scores: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Deep-copy score dict (CPU float)."""
    return scores_to_cpu(scores)


class _BaseSignal:
    """Shared bookkeeping for concrete local signals."""

    name: str

    def __init__(self, decay: float = DEFAULT_DECAY) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self._step: int | None = None
        self._neuron: dict[str, Tensor] = {}
        self._edge: dict[str, Tensor] = {}
        self._allocated: bool = False

    def reset(self) -> None:
        self._step = None
        self._neuron.clear()
        self._edge.clear()
        self._allocated = False
        self._on_reset()

    def _on_reset(self) -> None:
        """Subclass hook for extra state."""

    def neuron_scores(self) -> dict[str, Tensor]:
        return scores_to_cpu(self._neuron)

    def edge_scores(self) -> dict[str, Tensor]:
        return scores_to_cpu(self._edge)

    def snapshot_scores(self) -> ScoreBundle:
        return ScoreBundle(
            name=self.name,
            neuron=snapshot_dict(self._neuron),
            edge=snapshot_dict(self._edge),
            step=self._step,
        )

    def _ensure_buffers(
        self,
        layer_shapes: Mapping[str, tuple[int, int]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if self._allocated:
            # Fixed-topology contract: layer key set must match first allocation.
            allocated = set(self._neuron.keys())
            requested = set(layer_shapes.keys())
            if allocated != requested:
                raise ConfigError(
                    f"{self.name}: layer_shapes keys changed after allocation; "
                    f"allocated={sorted(allocated)} requested={sorted(requested)}. "
                    f"Pass the full fixed layer map on every update (or call reset())."
                )
            return
        for layer, (n_out, n_in) in layer_shapes.items():
            self._neuron[layer] = torch.zeros(n_out, device=device, dtype=dtype)
            self._edge[layer] = torch.zeros(n_out, n_in, device=device, dtype=dtype)
        self._allocated = True
        self._on_allocate(layer_shapes, device=device, dtype=dtype)

    def _on_allocate(
        self,
        layer_shapes: Mapping[str, tuple[int, int]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Subclass hook after score buffers are created."""
