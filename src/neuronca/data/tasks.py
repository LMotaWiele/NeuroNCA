"""Task transforms and shared TaskContext for train/probe loaders."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor


class TaskTransform(Protocol):
    def map_input(self, x: Tensor) -> Tensor: ...

    def map_label(self, y: Tensor) -> Tensor: ...


class IdentityTransform:
    """Task A: leave inputs and labels unchanged."""

    def map_input(self, x: Tensor) -> Tensor:
        return x

    def map_label(self, y: Tensor) -> Tensor:
        return y


class LabelPermutationTransform:
    """Task B1: apply a fixed permutation π to class labels."""

    def __init__(self, permutation: Tensor) -> None:
        if permutation.ndim != 1:
            raise ValueError("permutation must be a 1-D tensor of class indices")
        self.permutation = permutation.long().clone()

    def map_input(self, x: Tensor) -> Tensor:
        return x

    def map_label(self, y: Tensor) -> Tensor:
        return self.permutation[y.long()]


class DigitSplitTransform:
    """B2: remap labels in [low, high] to 0..K-1. Filter the dataset first."""

    def __init__(self, low: int = 5, high: int = 9) -> None:
        if low > high:
            raise ValueError("low must be <= high")
        self.low = int(low)
        self.high = int(high)

    @property
    def num_classes(self) -> int:
        return self.high - self.low + 1

    def map_input(self, x: Tensor) -> Tensor:
        return x

    def map_label(self, y: Tensor) -> Tensor:
        y = y.long()
        # Accept 0-d or 1-d; validate all elements are in range.
        if ((y < self.low) | (y > self.high)).any().item():
            bad = y[((y < self.low) | (y > self.high))]
            raise ValueError(
                f"DigitSplitTransform expects labels in [{self.low}, {self.high}]; "
                f"got out-of-range value(s) e.g. {int(bad.reshape(-1)[0])}. "
                "Filter the dataset with indices_in_digit_range before applying B2."
            )
        return y - self.low


class PixelPermutationTransform:
    """Task B3: apply a fixed pixel permutation (permuted MNIST)."""

    def __init__(self, permutation: Tensor) -> None:
        if permutation.ndim != 1:
            raise ValueError("permutation must be a 1-D index tensor")
        self.permutation = permutation.long().clone()
        self.num_pixels = int(self.permutation.numel())

    def map_input(self, x: Tensor) -> Tensor:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        perm = self.permutation.to(device=x.device)

        # Flatten trailing spatial dims of size num_pixels, apply perm, reshape.
        if x.shape[-1] == self.num_pixels and x.ndim >= 1:
            # Already flat on last dim: (..., N)
            return x[..., perm]

        if x.ndim >= 2 and x.shape[-1] * x.shape[-2] == self.num_pixels:
            # Spatial (..., H, W) with H*W == num_pixels
            leading = x.shape[:-2]
            h, w = x.shape[-2], x.shape[-1]
            flat = x.reshape(*leading, h * w)
            return flat[..., perm].reshape(*leading, h, w)

        if x.ndim == 1 and x.numel() == self.num_pixels:
            return x[perm]

        raise ValueError(
            f"unsupported input shape for pixel permutation: {tuple(x.shape)} "
            f"(expected last spatial dims product {self.num_pixels})"
        )

    def map_label(self, y: Tensor) -> Tensor:
        return y


class TaskContext:
    """Shared active transform for train and probe views.

    ``set_transform`` is in-process only. With ``num_workers>0``, drain iterators
    and rebuild loaders at A→B — workers will not see a mid-lifetime assignment.
    See ``docs/design/invariants.md``.
    """

    def __init__(self, transform: TaskTransform | None = None) -> None:
        self._transform: TaskTransform = (
            transform if transform is not None else IdentityTransform()
        )
        self._version: int = 0

    @property
    def transform(self) -> TaskTransform:
        return self._transform

    @property
    def version(self) -> int:
        """Monotonic counter; increments on each ``set_transform`` call."""
        return self._version

    def set_transform(self, t: TaskTransform) -> None:
        """Atomically replace the active transform for all in-process consumers.

        See class docstring for multi-worker DataLoader caveats.
        """
        self._transform = t
        self._version += 1

    def map_input(self, x: Tensor) -> Tensor:
        return self._transform.map_input(x)

    def map_label(self, y: Tensor) -> Tensor:
        return self._transform.map_label(y)


def indices_in_digit_range(
    targets: Tensor,
    low: int = 5,
    high: int = 9,
) -> Tensor:
    """Return indices whose labels lie in ``[low, high]`` (for Task B2).

    Use this to subset train/probe/test before applying
    :class:`DigitSplitTransform`. Design B2 is a restricted 5-way digit split
    (e.g. A: 0–4, B: 5–9), not a full-MNIST label shift.
    """
    if low > high:
        raise ValueError("low must be <= high")
    targets = targets.long()
    mask = (targets >= low) & (targets <= high)
    return torch.nonzero(mask, as_tuple=False).view(-1).long()


def make_identity_transform() -> IdentityTransform:
    return IdentityTransform()


def make_label_permutation_transform(
    num_classes: int = 10,
    seed: int = 123,
) -> LabelPermutationTransform:
    """Factory for B1: seeded label permutation π over ``num_classes``."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = torch.randperm(num_classes, generator=g)
    return LabelPermutationTransform(perm)


def make_digit_split_transform(low: int = 5, high: int = 9) -> DigitSplitTransform:
    """Factory for B2: digit-range split (default 5–9).

    Does **not** filter the dataset. Pair with :func:`indices_in_digit_range`
    (and a 5-way head) before training; out-of-range labels raise in
    ``map_label``.
    """
    return DigitSplitTransform(low=low, high=high)


def make_pixel_permutation_transform(
    num_pixels: int = 28 * 28,
    seed: int = 123,
) -> PixelPermutationTransform:
    """Factory for B3: fixed pixel shuffle seeded by ``seed``."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = torch.randperm(num_pixels, generator=g)
    return PixelPermutationTransform(perm)


def make_task_transform(
    shift_type: str,
    *,
    seed: int = 123,
    num_classes: int = 10,
) -> TaskTransform:
    """Build a task transform from a config-style shift type name."""
    key = shift_type.lower().replace("-", "_")
    if key in ("identity", "a", "task_a"):
        return make_identity_transform()
    if key in ("label_permutation", "b1"):
        return make_label_permutation_transform(num_classes=num_classes, seed=seed)
    if key in ("digit_split", "b2"):
        return make_digit_split_transform(low=5, high=9)
    if key in ("pixel_permutation", "b3"):
        return make_pixel_permutation_transform(num_pixels=28 * 28, seed=seed)
    raise ValueError(f"unknown shift_type: {shift_type!r}")
