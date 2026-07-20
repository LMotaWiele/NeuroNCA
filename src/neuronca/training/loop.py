"""Standard backprop training loop (CE + optimizer step; no local signals)."""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Return ``cuda`` if available else ``cpu``, or the explicit device."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _unpack_batch(batch: Any) -> tuple[Tensor, Tensor]:
    """Unpack ``(x, y)`` or ``(x, y, index)`` from a DataLoader batch."""
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError(
            f"expected batch as (x, y) or (x, y, ...); got {type(batch)!r}"
        )
    x, y = batch[0], batch[1]
    return x, y


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    *,
    device: str | torch.device | None = None,
    criterion: nn.Module | None = None,
    max_steps: int | None = None,
) -> dict[str, float]:
    """Run one training epoch (or up to ``max_steps`` optimizer steps).

    Uses cross-entropy on logits; no local signals or structure changes.

    Returns:
        Dict with mean ``loss`` and ``acc`` over processed examples.
    """
    device_t = resolve_device(device)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.train()
    model.to(device_t)

    total_loss = 0.0
    total_correct = 0
    total_n = 0
    steps = 0

    for batch in loader:
        if max_steps is not None and steps >= max_steps:
            break
        x, y = _unpack_batch(batch)
        x = x.to(device_t, non_blocking=True)
        y = y.to(device_t, non_blocking=True)
        if y.ndim > 1:
            y = y.view(-1)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = int(y.size(0))
        total_loss += float(loss.item()) * bs
        preds = logits.argmax(dim=1)
        total_correct += int((preds == y).sum().item())
        total_n += bs
        steps += 1

    if total_n == 0:
        return {"loss": float("nan"), "acc": float("nan"), "n": 0.0, "steps": 0.0}

    return {
        "loss": total_loss / total_n,
        "acc": total_correct / total_n,
        "n": float(total_n),
        "steps": float(steps),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: str | torch.device | None = None,
    criterion: nn.Module | None = None,
    max_steps: int | None = None,
) -> dict[str, float]:
    """Evaluate mean CE loss and accuracy (model left in eval mode).

    Returns:
        Dict with mean ``loss`` and ``acc`` over processed examples.
    """
    device_t = resolve_device(device)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    model.to(device_t)

    total_loss = 0.0
    total_correct = 0
    total_n = 0
    steps = 0

    for batch in loader:
        if max_steps is not None and steps >= max_steps:
            break
        x, y = _unpack_batch(batch)
        x = x.to(device_t, non_blocking=True)
        y = y.to(device_t, non_blocking=True)
        if y.ndim > 1:
            y = y.view(-1)

        logits = model(x)
        loss = criterion(logits, y)

        bs = int(y.size(0))
        total_loss += float(loss.item()) * bs
        preds = logits.argmax(dim=1)
        total_correct += int((preds == y).sum().item())
        total_n += bs
        steps += 1

    if total_n == 0:
        return {"loss": float("nan"), "acc": float("nan"), "n": 0.0, "steps": 0.0}

    return {
        "loss": total_loss / total_n,
        "acc": total_correct / total_n,
        "n": float(total_n),
        "steps": float(steps),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    *,
    epochs: int = 1,
    device: str | torch.device | None = None,
    criterion: nn.Module | None = None,
    eval_loader: DataLoader | None = None,
    max_steps_per_epoch: int | None = None,
    epoch_callback: Callable[[int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train for ``epochs`` with CE + optimizer; optionally evaluate each epoch.

    No local signals, no structure changes — pure backprop.

    Args:
        model: Network to train (moved to ``device``).
        train_loader: Training DataLoader yielding ``(x, y)`` batches.
        optimizer: Optimizer from :func:`~neuronca.training.optim.build_optimizer`.
        epochs: Number of full epochs (or partial if ``max_steps_per_epoch`` set).
        device: Target device; ``None``/``"auto"`` → cuda if available else cpu.
        criterion: Loss module; defaults to ``CrossEntropyLoss``.
        eval_loader: Optional loader for per-epoch evaluation metrics.
        max_steps_per_epoch: Cap optimizer steps per epoch (smoke / debug).
        epoch_callback: Optional ``(epoch_idx, metrics)`` hook after each epoch.

    Returns:
        Dict with:
        - ``history``: list of per-epoch metric dicts
        - ``train_loss`` / ``train_acc``: final epoch train metrics
        - ``eval_loss`` / ``eval_acc``: final eval metrics if ``eval_loader`` given
        - ``device``: resolved device string
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1; got {epochs}")

    device_t = resolve_device(device)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.to(device_t)
    history: list[dict[str, Any]] = []

    for epoch in range(epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device_t,
            criterion=criterion,
            max_steps=max_steps_per_epoch,
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_n": train_metrics["n"],
            "train_steps": train_metrics["steps"],
        }
        if eval_loader is not None:
            eval_metrics = evaluate(
                model,
                eval_loader,
                device=device_t,
                criterion=criterion,
            )
            row["eval_loss"] = eval_metrics["loss"]
            row["eval_acc"] = eval_metrics["acc"]
            row["eval_n"] = eval_metrics["n"]

        history.append(row)
        if epoch_callback is not None:
            epoch_callback(epoch, row)

    result: dict[str, Any] = {
        "history": history,
        "train_loss": history[-1]["train_loss"],
        "train_acc": history[-1]["train_acc"],
        "device": str(device_t),
        "epochs": epochs,
    }
    if eval_loader is not None:
        result["eval_loss"] = history[-1]["eval_loss"]
        result["eval_acc"] = history[-1]["eval_acc"]
    return result
