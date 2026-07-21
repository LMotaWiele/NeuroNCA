"""Shared unit-square coordinates for Phase 1a locality (sensing radius R).

Coordinates are identity of a *living* neuron: assigned at birth, immutable while
active, discarded on prune, fresh draw on slot reactivation.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def input_coords_28x28(device: torch.device | None = None) -> Tensor:
    """Map 784 MNIST pixels to unit-square coords on a 28×28 grid.

    Returns:
        Float tensor ``[784, 2]`` with (x, y) in ``[0, 1]``.
    """
    # Pixel (row, col) → (x, y) with x = col/27, y = row/27.
    rows = torch.arange(28, dtype=torch.float32, device=device)
    cols = torch.arange(28, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    coords = torch.stack([xx.reshape(-1) / 27.0, yy.reshape(-1) / 27.0], dim=-1)
    return coords


def output_coords(num_classes: int = 10, device: torch.device | None = None) -> Tensor:
    """Place output units evenly along the bottom of the unit square.

    Returns:
        Float tensor ``[num_classes, 2]``.
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    if num_classes == 1:
        xs = torch.tensor([0.5], dtype=torch.float32, device=device)
    else:
        xs = torch.linspace(0.0, 1.0, num_classes, dtype=torch.float32, device=device)
    ys = torch.zeros(num_classes, dtype=torch.float32, device=device)
    return torch.stack([xs, ys], dim=-1)


def assign_hidden_coord(
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
    *,
    existing: Tensor | None = None,
    min_sep: float = 0.0,
    max_tries: int = 64,
) -> Tensor:
    """Draw a fresh unit-square coordinate for a newly born hidden neuron.

    Args:
        generator: Optional RNG for reproducibility.
        device: Device for the returned tensor.
        existing: Optional ``[N, 2]`` active coords to avoid (soft uniqueness).
        min_sep: Minimum Euclidean separation from ``existing`` when possible.
        max_tries: Rejection-sampling attempts when ``min_sep > 0``.

    Returns:
        Float tensor ``[2]`` in ``[0, 1]^2``.
    """
    for _ in range(max_tries):
        coord = torch.rand(2, generator=generator, device=device, dtype=torch.float32)
        if existing is None or existing.numel() == 0 or min_sep <= 0:
            return coord
        d = torch.linalg.norm(existing - coord.unsqueeze(0), dim=-1)
        if bool((d >= min_sep).all()):
            return coord
    # Fall through: return last draw even if separation failed.
    return torch.rand(2, generator=generator, device=device, dtype=torch.float32)


def pairwise_distances(a: Tensor, b: Tensor) -> Tensor:
    """Euclidean distances between point sets ``a[N,2]`` and ``b[M,2]`` → ``[N,M]``."""
    if a.ndim != 2 or a.shape[-1] != 2:
        raise ValueError(f"a must be [N,2], got {tuple(a.shape)}")
    if b.ndim != 2 or b.shape[-1] != 2:
        raise ValueError(f"b must be [M,2], got {tuple(b.shape)}")
    # (N,1,2) - (1,M,2) → (N,M,2)
    return torch.linalg.norm(a.unsqueeze(1) - b.unsqueeze(0), dim=-1)


def linear_decay(distance: Tensor, radius: float) -> Tensor:
    """Monotonic linear decay on ``[0, R]``: ``1 - d/R``; zeros beyond R.

    Args:
        distance: Non-negative distances (any shape).
        radius: Sensing radius ``R > 0``.

    Returns:
        Same shape as ``distance``; values in ``[0, 1]`` for ``d <= R``, else 0.
    """
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    decay = 1.0 - distance / radius
    return torch.clamp(decay, min=0.0)
