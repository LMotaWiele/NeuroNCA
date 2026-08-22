"""Snapshot-scoped probe tensor cache for GT measures.

Never reuse across ``set_transform`` / snapshot boundaries
(``docs/design/invariants.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from neuronca.data.tasks import TaskContext

# Probe source accepted by GT helpers (live loader or snapshot cache).
ProbeSource = Union[DataLoader, "ProbeCache"]


def _unpack_batch(batch: Any) -> tuple[Tensor, Tensor]:
    """Local unpack to avoid circular import with ``importance.base``."""
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError(
            f"expected batch as (x, y) or (x, y, ...); got {type(batch)!r}"
        )
    return batch[0], batch[1]


class ProbeCacheVersionError(RuntimeError):
    """Raised when a cache's stored task version does not match the live context."""


class ProbeCacheIntegrityError(RuntimeError):
    """Raised when cached probe tensors no longer match a fresh loader materialization."""


@dataclass
class ProbeCache:
    """Materialized probe tensors for one snapshot under one task version.

    Attributes:
        X: Inputs, on-device, detached contiguous clone ``(N, ...)``.
        Y: Labels, on-device, detached contiguous clone ``(N,)`` or ``(N, ...)``.
        task_version: ``TaskContext.version`` at materialization time, or ``None``
            when no task context is available (toy TensorDataset tests).
        batch_size: Mini-batch size used when iterating the cache.
        task_context: Optional live context for version guards on every use.
    """

    X: Tensor
    Y: Tensor
    task_version: int | None
    batch_size: int
    task_context: Any | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1; got {self.batch_size}")
        if self.X.size(0) != self.Y.size(0):
            raise ValueError(
                f"X/Y length mismatch: X={self.X.size(0)} Y={self.Y.size(0)}"
            )

    def __len__(self) -> int:
        return int(self.X.size(0))

    def ensure_version(self) -> None:
        """Fail loud if the live TaskContext version diverged from build time.

        No-op when no task context was bound (unit tests with TensorDataset).
        """
        ctx = self.task_context
        if ctx is None or self.task_version is None:
            return
        current = int(ctx.version)
        if current != int(self.task_version):
            raise ProbeCacheVersionError(
                f"ProbeCache is stale: built at task_version={self.task_version}, "
                f"current TaskContext.version={current}. Rebuild the cache at the "
                "start of each snapshot; never reuse across TaskContext.set_transform."
            )

    def iter_batches(self) -> Iterator[tuple[Tensor, Tensor]]:
        """Yield ``(x, y)`` mini-batches from the cached tensors (version-checked)."""
        self.ensure_version()
        n = len(self)
        bs = self.batch_size
        for start in range(0, n, bs):
            end = min(start + bs, n)
            yield self.X[start:end], self.Y[start:end]


def extract_task_context(probe_loader: DataLoader) -> Any | None:
    """Return ``dataset.task_context`` when the loader wraps a TaskAwareDataset."""
    ds: Any = probe_loader.dataset
    # Unwrap Subset-like shells if present (TaskAwareDataset is usually outer).
    seen: set[int] = set()
    while ds is not None and id(ds) not in seen:
        seen.add(id(ds))
        ctx = getattr(ds, "task_context", None)
        if ctx is not None and hasattr(ctx, "version"):
            return ctx
        ds = getattr(ds, "dataset", None) or getattr(ds, "base", None)
    return None


@torch.no_grad()
def materialize_probe_cache(
    probe_loader: DataLoader,
    *,
    device: torch.device | str,
    batch_size: int | None = None,
    task_context: Any | None = None,
) -> ProbeCache:
    """Run one full probe pass and return post-transform tensors on ``device``.

    Tensors are ``.detach().contiguous().clone()`` so the cache owns its storage
    and is isolated from future loader / collate buffers.
    """
    device = torch.device(device)
    if task_context is None:
        task_context = extract_task_context(probe_loader)
    task_version = int(task_context.version) if task_context is not None else None

    xs: list[Tensor] = []
    ys: list[Tensor] = []
    for batch in probe_loader:
        x, y = _unpack_batch(batch)
        xs.append(x.detach())
        ys.append(y.detach())
    if not xs:
        raise ValueError("probe_loader yielded no examples; cannot materialize cache")

    X = torch.cat(xs, dim=0).to(device).detach().contiguous().clone()
    Y = torch.cat(ys, dim=0).to(device).detach().contiguous().clone()

    bs = batch_size
    if bs is None:
        lb = getattr(probe_loader, "batch_size", None)
        bs = int(lb) if lb is not None and int(lb) > 0 else 128

    return ProbeCache(
        X=X,
        Y=Y,
        task_version=task_version,
        batch_size=int(bs),
        task_context=task_context,
    )


@torch.no_grad()
def assert_probe_cache_matches_loader(
    cache: ProbeCache,
    probe_loader: DataLoader,
    *,
    device: torch.device | str | None = None,
) -> None:
    """Integrity guard: cache tensors must equal a fresh materialization.

    Runs after the GT suite within a snapshot. Catches in-place corruption of
    shared storage (e.g. future hooks writing through ``x.view``).
    """
    cache.ensure_version()
    dev = torch.device(device) if device is not None else cache.X.device
    fresh = materialize_probe_cache(
        probe_loader,
        device=dev,
        batch_size=cache.batch_size,
        task_context=cache.task_context,
    )
    if not torch.equal(cache.X, fresh.X):
        raise ProbeCacheIntegrityError(
            "ProbeCache.X diverged from a fresh probe_loader materialization after "
            "the GT suite — cached inputs were mutated or the loader is no longer "
            "deterministic under the current TaskContext."
        )
    if not torch.equal(cache.Y, fresh.Y):
        raise ProbeCacheIntegrityError(
            "ProbeCache.Y diverged from a fresh probe_loader materialization after "
            "the GT suite — cached labels were mutated or the loader is no longer "
            "deterministic under the current TaskContext."
        )


def iter_probe_batches(
    source: ProbeSource,
    *,
    device: torch.device | None = None,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Iterate ``(x, y)`` from a DataLoader or :class:`ProbeCache`.

    Cached batches are already on-device; loader batches are moved to ``device``
    when provided.
    """
    if isinstance(source, ProbeCache):
        yield from source.iter_batches()
        return

    for batch in source:
        x, y = _unpack_batch(batch)
        if device is not None:
            x = x.to(device)
            y = y.to(device)
        yield x, y
