"""One-epoch ActivationMagnitude probe → per-layer r*_ℓ warm-start."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from neuronca.broadcast.error import BroadcastPacket
from neuronca.models.dynamic_mlp import HIDDEN_LAYERS, DynamicMLPReasoningNet
from neuronca.signals.activation import ActivationMagnitude
from neuronca.training.loop import resolve_device
from neuronca.training.optim import build_optimizer

LIN_FOR = {"h1": "fc1", "h2": "fc2", "h3": "fc3"}


def _dummy_broadcast(batch_size: int, num_classes: int, step: int) -> BroadcastPacket:
    return BroadcastPacket(
        e_scalar_ce=0.0,
        e_vec_mean=torch.zeros(num_classes),
        e_s_per_sample=torch.zeros(batch_size),
        r_per_sample=torch.zeros(batch_size, num_classes),
        step=step,
    )


@torch.no_grad()
def active_mean_ema_rates(
    signal: ActivationMagnitude,
    net: DynamicMLPReasoningNet,
) -> dict[str, float]:
    """Mean EMA activation among active units per hidden layer → r*_ℓ."""
    scores = signal.neuron_scores()
    out: dict[str, float] = {}
    for h, lin in LIN_FOR.items():
        key = lin if lin in scores else h
        if key not in scores:
            raise KeyError(f"no neuron scores for {h}/{lin}; keys={list(scores)}")
        s = scores[key].detach().float().cpu()
        mask = net.active_mask(h).detach().cpu().bool()
        n = min(s.numel(), mask.numel())
        s, mask = s[:n], mask[:n]
        active = s[mask]
        if active.numel() == 0:
            raise RuntimeError(f"no active units in {h} for rate probe")
        out[h] = float(active.mean().item())
    return out


def probe_target_rates(
    net: DynamicMLPReasoningNet,
    train_loader: DataLoader,
    *,
    signal: ActivationMagnitude | None = None,
    optimizer: Optimizer | None = None,
    device: str | torch.device | None = None,
    epochs: int = 1,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
) -> dict[str, Any]:
    """Train ``epochs`` without plasticity; return layer mean EMA rates + stats.

    This is the warm-start for self-calibrating r*_ℓ — same protocol as the
    historical one-epoch target-rate probe (MNIST/Fashion).
    """
    import torch.nn as nn

    device_t = resolve_device(device)
    net = net.to(device_t)
    if signal is None:
        signal = ActivationMagnitude(decay=0.99)
    if optimizer is None:
        optimizer = build_optimizer(
            net, "sgd", lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    criterion = nn.CrossEntropyLoss()
    step = 0
    total_loss = 0.0
    total_n = 0
    net.train()
    for _ep in range(int(epochs)):
        for batch in train_loader:
            x, y = batch[0], batch[1]
            x = x.to(device_t, non_blocking=True)
            y = y.to(device_t, non_blocking=True)
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            if y.ndim > 1:
                y = y.view(-1)
            optimizer.zero_grad(set_to_none=True)
            logits = net(x)
            loss = criterion(logits, y)
            loss.backward()
            if hasattr(net, "zero_inactive_grads"):
                net.zero_inactive_grads()
            optimizer.step()
            acts = net.last_activations
            assert acts is not None
            signal.update(
                activations=acts,
                layer_shapes=net.layer_shapes(),
                broadcast=_dummy_broadcast(int(y.size(0)), int(net.num_classes), step),
                step=step,
            )
            bs = int(y.size(0))
            total_loss += float(loss.item()) * bs
            total_n += bs
            step += 1
    rates = active_mean_ema_rates(signal, net)
    # Rounded product-style lock (2 decimals) optional for reporting
    rates_rounded = {h: round(rates[h], 2) for h in HIDDEN_LAYERS}
    return {
        "target_rates": rates,
        "target_rates_rounded": rates_rounded,
        "steps": step,
        "mean_loss": total_loss / max(total_n, 1),
        "signal": signal,
    }
