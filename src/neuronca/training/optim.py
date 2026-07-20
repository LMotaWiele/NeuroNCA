"""Optimizer factory for Phase 0 backprop training (SGD-first)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

# Frozen Phase 0 defaults (configs/default.yaml, design doc Training Defaults).
DEFAULT_OPTIMIZER = "sgd"
DEFAULT_LR = 0.01
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_ADAM_BETAS = (0.9, 0.999)
DEFAULT_ADAM_EPS = 1e-8


def build_optimizer(
    model: nn.Module,
    name: str = DEFAULT_OPTIMIZER,
    *,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    momentum: float = DEFAULT_MOMENTUM,
    betas: tuple[float, float] = DEFAULT_ADAM_BETAS,
    eps: float = DEFAULT_ADAM_EPS,
    **hparams: Any,
) -> Optimizer:
    """Build an optimizer for ``model`` parameters.

    Default is **SGD** (momentum 0.9, lr 0.01, weight_decay 1e-4). Adam is
    available via ``name="adam"`` but is not the Phase 0 default.

    Args:
        model: Module whose ``parameters()`` will be optimized.
        name: ``"sgd"`` (default) or ``"adam"`` (case-insensitive).
        lr: Learning rate.
        weight_decay: L2 weight decay.
        momentum: SGD momentum (ignored for Adam).
        betas: Adam beta coefficients (ignored for SGD).
        eps: Adam epsilon (ignored for SGD).
        **hparams: Extra kwargs forwarded to the torch optimizer constructor
            (must not collide with the explicit kwargs above).

    Returns:
        A ``torch.optim.Optimizer`` instance.

    Raises:
        ValueError: If ``name`` is not a supported optimizer.
    """
    key = name.strip().lower()
    params = model.parameters()

    if key == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            **hparams,
        )
    if key == "adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            **hparams,
        )
    raise ValueError(
        f"unsupported optimizer {name!r}; expected 'sgd' or 'adam'"
    )
