"""Training loop with per-batch activation-signal EMA and per-epoch plasticity."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from neuronca.broadcast.error import BroadcastPacket
from neuronca.models.dynamic_mlp import HIDDEN_LAYERS, DynamicMLPReasoningNet
from neuronca.plasticity.orchestrator import PlasticityOrchestrator, StructureEventLog
from neuronca.plasticity.rules import (
    GatedNoveltyContext,
    evaluate_gated_novelty_context,
)
from neuronca.plasticity.structure_snapshot import StructureSnapshotRecorder
from neuronca.signals.activation import ActivationMagnitude
from neuronca.training.loop import evaluate, resolve_device, train_epoch


def _dummy_broadcast(batch_size: int, num_classes: int, step: int) -> BroadcastPacket:
    """ActivationMagnitude ignores broadcast; provide a valid packet for the API."""
    return BroadcastPacket(
        e_scalar_ce=0.0,
        e_vec_mean=torch.zeros(num_classes),
        e_s_per_sample=torch.zeros(batch_size),
        r_per_sample=torch.zeros(batch_size, num_classes),
        step=step,
    )


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError(f"expected batch (x, y, ...), got {type(batch)!r}")
    return batch[0], batch[1]


def train_epoch_with_signals(
    model: DynamicMLPReasoningNet,
    loader: DataLoader,
    optimizer: Optimizer,
    signal: ActivationMagnitude,
    orch: PlasticityOrchestrator,
    *,
    device: str | torch.device | None = None,
    criterion: nn.Module | None = None,
    max_steps: int | None = None,
    global_step: int = 0,
) -> dict[str, float]:
    """One training epoch: CE + SGD, activation EMA, zero inactive grads."""
    device_t = resolve_device(device)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.train()
    model.to(device_t)

    total_loss = 0.0
    total_correct = 0
    total_n = 0
    steps = 0
    step = global_step

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
        model.zero_inactive_grads()
        # Critical-period analog: elevated effective LR on newborn incident
        # weights while age < grace (see PlasticityConfig.newborn_lr_mult).
        model.scale_newborn_critical_grads(
            lr_mult=float(orch.config.newborn_lr_mult),
            grace_steps=int(orch.config.newborn_grace_steps),
        )
        optimizer.step()

        bs = int(y.size(0))
        # Update local activation signal from detached activations.
        acts = model.last_activations
        assert acts is not None
        orch.update_input_signal_ema(acts["act_in"])
        signal.update(
            activations=acts,
            layer_shapes=model.layer_shapes(),
            broadcast=_dummy_broadcast(bs, model.num_classes, step),
            step=step,
        )

        total_loss += float(loss.item()) * bs
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_n += bs
        steps += 1
        step += 1

    if total_n == 0:
        return {
            "loss": float("nan"),
            "acc": float("nan"),
            "n": 0.0,
            "steps": 0.0,
            "global_step": float(step),
        }
    return {
        "loss": total_loss / total_n,
        "acc": total_correct / total_n,
        "n": float(total_n),
        "steps": float(steps),
        "global_step": float(step),
    }


@torch.no_grad()
def collect_gated_novelty_probe(
    model: DynamicMLPReasoningNet,
    loader: DataLoader,
    *,
    device: str | torch.device | None = None,
    max_batches: int | None = None,
    radius: float = 0.35,
    k_min: int = 11,
) -> GatedNoveltyContext:
    """Run a probe pass and evaluate the three-part gated-novelty rule.

    Collects per-sample CE, hidden post-activations, and incoming
    pre-activations (for gap_imprinted birth).
    """
    device_t = resolve_device(device)
    model.eval()
    model.to(device_t)

    act_bufs: dict[str, list[torch.Tensor]] = {h: [] for h in HIDDEN_LAYERS}
    pre_bufs: dict[str, list[torch.Tensor]] = {h: [] for h in HIDDEN_LAYERS}
    ce_chunks: list[torch.Tensor] = []
    act_key = {"h1": "act_fc1", "h2": "act_fc2", "h3": "act_fc3"}

    n_batches = 0
    for batch in loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        x, y = _unpack_batch(batch)
        x = x.to(device_t, non_blocking=True)
        y = y.to(device_t, non_blocking=True)
        if y.ndim > 1:
            y = y.view(-1)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        # Manual pre-acts for gap imprint (same path as DynamicMLP.forward).
        act_in = x
        pre1 = F.linear(
            act_in,
            model.effective_weight("fc1"),
            model.effective_bias("fc1"),
        )
        h1 = F.relu(pre1)
        pre2 = F.linear(
            h1, model.effective_weight("fc2"), model.effective_bias("fc2")
        )
        h2 = F.relu(pre2)
        pre3 = F.linear(
            h2, model.effective_weight("fc3"), model.effective_bias("fc3")
        )
        h3 = F.relu(pre3)
        logits = F.linear(
            h3, model.effective_weight("fc_out"), model.effective_bias("fc_out")
        )
        # Keep last_activations consistent with a normal forward.
        model.last_activations = {
            "act_in": act_in.detach().clone(),
            "act_fc1": h1.detach().clone(),
            "act_fc2": h2.detach().clone(),
            "act_fc3": h3.detach().clone(),
            "act_fc_out": logits.detach().clone(),
        }
        ce = F.cross_entropy(logits, y, reduction="none")
        ce_chunks.append(ce.detach().cpu())
        posts = {"h1": h1, "h2": h2, "h3": h3}
        pres = {"h1": pre1, "h2": pre2, "h3": pre3}
        for h in HIDDEN_LAYERS:
            act_bufs[h].append(posts[h].detach().cpu())
            # Pre-act for imprint uses the *incoming* drive: for gap residual we
            # want the pre-synaptic population activity (input to this layer),
            # not the post-ReLU pre of this unit. Incoming row maps pre→post:
            # h1 ← act_in, h2 ← h1, h3 ← h2.
            if h == "h1":
                pre_bufs[h].append(act_in.detach().cpu())
            elif h == "h2":
                pre_bufs[h].append(h1.detach().cpu())
            else:
                pre_bufs[h].append(h2.detach().cpu())
        n_batches += 1

    if not ce_chunks:
        # Empty probe → empty context (no growth triggers).
        return GatedNoveltyContext()

    ce_all = torch.cat(ce_chunks, dim=0)
    layer_acts = {h: torch.cat(act_bufs[h], dim=0) for h in HIDDEN_LAYERS}
    pre_acts = {h: torch.cat(pre_bufs[h], dim=0) for h in HIDDEN_LAYERS}
    return evaluate_gated_novelty_context(
        model,
        layer_acts,
        ce_all,
        radius=float(radius),
        k_min=int(k_min),
        pre_acts=pre_acts,
    )


def train_plastic_model(
    model: DynamicMLPReasoningNet,
    train_loader: DataLoader,
    optimizer: Optimizer,
    orch: PlasticityOrchestrator,
    *,
    signal: ActivationMagnitude | None = None,
    epochs: int = 20,
    device: str | torch.device | None = None,
    val_loader: DataLoader | None = None,
    max_steps_per_epoch: int | None = None,
    probe_loader: DataLoader | None = None,
    novelty_probe_max_batches: int | None = None,
    structure_snapshot_recorder: StructureSnapshotRecorder | None = None,
) -> dict[str, Any]:
    """Full multi-epoch plastic training with once-per-epoch structure update.

    When ``orch.config.neuron_growth_trigger == "gated_novelty"``, a probe
    loader is required (``probe_loader`` or ``val_loader``) to evaluate the
    three-part rule. Under-floor units are excluded and logged on the structure
    event (no silent fallback to act > r*).

    ``structure_snapshot_recorder`` is optional observability only (no effect
    on training when None / disabled).
    """
    if signal is None:
        signal = ActivationMagnitude(decay=0.99)

    history_train: list[dict[str, float]] = []
    history_val: list[dict[str, float]] = []
    structure: list[StructureEventLog] = []
    snapshot_paths: list[str] = []
    global_step = 0
    trigger_mode = str(orch.config.neuron_growth_trigger).strip().lower()
    use_gated = trigger_mode in ("gated_novelty", "novelty", "gated")
    gate_loader = probe_loader if probe_loader is not None else val_loader
    if use_gated and gate_loader is None:
        raise ValueError(
            "gated_novelty growth requires probe_loader or val_loader for "
            "CE-gated evaluation; refusing to fall back to activation trigger"
        )

    for epoch in range(epochs):
        metrics = train_epoch_with_signals(
            model,
            train_loader,
            optimizer,
            signal,
            orch,
            device=device,
            max_steps=max_steps_per_epoch,
            global_step=global_step,
        )
        global_step = int(metrics["global_step"])
        history_train.append({"epoch": float(epoch), **metrics})

        gated_ctx: GatedNoveltyContext | None = None
        if use_gated:
            assert gate_loader is not None
            gated_ctx = collect_gated_novelty_probe(
                model,
                gate_loader,
                device=device,
                max_batches=novelty_probe_max_batches,
                radius=float(orch.config.radius),
                k_min=int(orch.config.novelty_k_min),
            )

        # Plasticity step (cadence = epoch).
        scores = signal.neuron_scores()
        slog = orch.step(
            model,
            scores,
            optimizer=optimizer,
            epoch=epoch,
            signal=signal,
            gated_novelty=gated_ctx,
        )
        structure.append(slog)

        if structure_snapshot_recorder is not None:
            path = structure_snapshot_recorder.maybe_record(
                epoch,
                model,
                signal=signal,
                structure_log=slog,
            )
            if path is not None:
                snapshot_paths.append(str(path))

        if val_loader is not None:
            vm = evaluate(model, val_loader, device=device)
            history_val.append({"epoch": float(epoch), **vm})

    return {
        "train": history_train,
        "val": history_val,
        "structure": [s.as_dict() for s in structure],
        "orchestrator": orch,
        "signal": signal,
        "structure_snapshots": snapshot_paths,
    }


# Re-export evaluate for callers
__all__ = [
    "train_epoch_with_signals",
    "train_plastic_model",
    "collect_gated_novelty_probe",
    "evaluate",
    "train_epoch",
]
