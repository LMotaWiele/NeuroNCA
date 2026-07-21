"""Instrumented training callbacks with hard snapshot isolation.

Train-step protocol
-------------------
On each TRAIN step:

1. Forward (activations collected on the model).
2. ``BroadcastErrorEncoder.encode`` → residual packet.
3. ``LocalSignal.update`` for each candidate (under ``no_grad``; never
   contributes to loss / optimizer).
4. Normal backprop: ``loss.backward()`` + ``optimizer.step()``.

Snapshot isolation protocol
---------------------------
On SNAPSHOT:

1. Freeze / copy local scores via ``snapshot_scores`` under ``no_grad``.
2. Run all GT measures with signals **disabled** (no ``LocalSignal.update``
   on probe forwards — GT modules must not call update; this module also
   flips an enable flag so instrumented hooks stay inert).
3. Resume training (re-enable signal updates).

Hard rule: trace tensors are bitwise-identical before and after a GT-only call.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from neuronca.broadcast.error import BroadcastErrorEncoder, BroadcastPacket
from neuronca.importance.base import GroundTruthImportance, ImportanceResult
from neuronca.signals.base import LocalSignal, ScoreBundle
from neuronca.training.loop import _unpack_batch, resolve_device


# ---------------------------------------------------------------------------
# Trace capture helpers (bitwise identity)
# ---------------------------------------------------------------------------


def _clone_tensor_tree(obj: Any) -> Any:
    """Deep-clone nested dict/list/tuple of tensors (detach+clone); pass through rest."""
    if torch.is_tensor(obj):
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {k: _clone_tensor_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clone_tensor_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_clone_tensor_tree(v) for v in obj)
    return obj


def capture_signal_traces(signal: LocalSignal) -> dict[str, Any]:
    """Capture live EMA / score buffers for bitwise isolation checks.

    Prefers well-known private buffers when present (``_T_edge``, ``_edge``,
    ``_neuron``, hebbian ``_m_*``). Falls back to exported edge/neuron scores.
    """
    state: dict[str, Any] = {"name": getattr(signal, "name", type(signal).__name__)}

    # Public scores (always available via protocol).
    state["edge_scores"] = {
        k: v.detach().clone() for k, v in signal.edge_scores().items()
    }
    state["neuron_scores"] = {
        k: v.detach().clone() for k, v in signal.neuron_scores().items()
    }

    # Internal buffers that accumulate across train steps (must not change under GT).
    for attr in (
        "_T_edge",
        "_edge",
        "_neuron",
        "_m_e",
        "_m_a_neuron",
        "_m_ae_neuron",
        "_m_a_edge",
        "_m_ae_edge",
        "_neuron_out_mean",
        "_step",
    ):
        if hasattr(signal, attr):
            state[attr] = _clone_tensor_tree(getattr(signal, attr))

    return state


def traces_bitwise_equal(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    keys: Sequence[str] | None = None,
) -> bool:
    """Return True if captured traces are bitwise-identical (tensor ``equal``)."""
    check_keys = list(keys) if keys is not None else [
        k for k in before.keys() if k != "name"
    ]
    for key in check_keys:
        if key not in before or key not in after:
            return False
        if not _values_bitwise_equal(before[key], after[key]):
            return False
    return True


def _values_bitwise_equal(a: Any, b: Any) -> bool:
    if torch.is_tensor(a) and torch.is_tensor(b):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        return bool(torch.equal(a, b))
    if type(a) is not type(b):
        # Allow int/None for _step
        return a == b
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_values_bitwise_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_values_bitwise_equal(x, y) for x, y in zip(a, b))
    return a == b


def assert_traces_bitwise_equal(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    keys: Sequence[str] | None = None,
    msg: str = "signal traces changed across GT-only call (snapshot isolation violated)",
) -> None:
    """Raise ``AssertionError`` unless traces match bitwise."""
    check_keys = list(keys) if keys is not None else [
        k for k in before.keys() if k != "name"
    ]
    for key in check_keys:
        if key not in before:
            raise AssertionError(f"{msg}: missing key {key!r} in before")
        if key not in after:
            raise AssertionError(f"{msg}: missing key {key!r} in after")
        if not _values_bitwise_equal(before[key], after[key]):
            raise AssertionError(f"{msg}: mismatch at key {key!r}")


# ---------------------------------------------------------------------------
# Training RNG isolation (GT must not advance the train-stream RNG)
# ---------------------------------------------------------------------------


class TrainingRNGIsolationError(RuntimeError):
    """Raised when GT/snapshot leaves torch RNG state different from pre-GT."""


def capture_torch_rng_state() -> dict[str, Tensor | list[Tensor] | None]:
    """Snapshot torch CPU (+ CUDA if present) RNG state for later restore."""
    state: dict[str, Tensor | list[Tensor] | None] = {
        "cpu": torch.get_rng_state().cpu().clone(),
    }
    if torch.cuda.is_available():
        # Device-local + all-device copies for multi-GPU safety.
        state["cuda"] = torch.cuda.get_rng_state().cpu().clone()
        state["cuda_all"] = [s.cpu().clone() for s in torch.cuda.get_rng_state_all()]
    else:
        state["cuda"] = None
        state["cuda_all"] = None
    return state


def restore_torch_rng_state(state: Mapping[str, Any]) -> None:
    """Restore torch RNG from :func:`capture_torch_rng_state`."""
    cpu = state["cpu"]
    if not isinstance(cpu, Tensor):
        raise TypeError(f"expected Tensor cpu RNG state; got {type(cpu)!r}")
    torch.set_rng_state(cpu)
    if torch.cuda.is_available() and state.get("cuda_all") is not None:
        torch.cuda.set_rng_state_all(list(state["cuda_all"]))
    elif torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"])


def assert_torch_rng_state_equal(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    msg: str = "torch RNG state changed across GT/snapshot (training RNG isolation violated)",
) -> None:
    """Raise :class:`TrainingRNGIsolationError` unless CPU (+ CUDA) RNG match."""
    b_cpu, a_cpu = before.get("cpu"), after.get("cpu")
    if not isinstance(b_cpu, Tensor) or not isinstance(a_cpu, Tensor):
        raise TrainingRNGIsolationError(f"{msg}: missing cpu RNG tensors")
    if not torch.equal(b_cpu, a_cpu):
        raise TrainingRNGIsolationError(f"{msg}: cpu RNG state mismatch")
    if torch.cuda.is_available():
        b_all, a_all = before.get("cuda_all"), after.get("cuda_all")
        if b_all is not None and a_all is not None:
            if len(b_all) != len(a_all) or any(
                not torch.equal(x, y) for x, y in zip(b_all, a_all)
            ):
                raise TrainingRNGIsolationError(f"{msg}: cuda RNG state mismatch")
        else:
            b_c, a_c = before.get("cuda"), after.get("cuda")
            if isinstance(b_c, Tensor) and isinstance(a_c, Tensor):
                if not torch.equal(b_c, a_c):
                    raise TrainingRNGIsolationError(f"{msg}: cuda RNG state mismatch")


# ---------------------------------------------------------------------------
# Snapshot result
# ---------------------------------------------------------------------------


@dataclass
class SnapshotResult:
    """Frozen local scores + GT results from an isolated snapshot."""

    step: int
    local_scores: dict[str, ScoreBundle]  # signal name → frozen scores
    gt_results: dict[str, ImportanceResult]  # gt name → result
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Instrumented trainer
# ---------------------------------------------------------------------------


class InstrumentedTrainer:
    """Backprop trainer that observes local signals without feeding them to loss.

    Signal updates are gated by :attr:`signals_enabled`. Snapshots disable the
    gate so probe / GT forwards never call ``LocalSignal.update`` through this
    trainer (GT modules also must not call update themselves).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        signals: Sequence[LocalSignal],
        *,
        encoder: BroadcastErrorEncoder | None = None,
        criterion: nn.Module | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.signals: list[LocalSignal] = list(signals)
        self.encoder = encoder if encoder is not None else BroadcastErrorEncoder()
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.device = resolve_device(device)
        self.model.to(self.device)

        self._signals_enabled: bool = True
        self.global_step: int = 0
        self._last_packet: BroadcastPacket | None = None
        self._last_loss: float | None = None

    # -- enable / disable ----------------------------------------------------

    @property
    def signals_enabled(self) -> bool:
        """When False, train-step and hooks skip ``LocalSignal.update``."""
        return self._signals_enabled

    def enable_signals(self) -> None:
        self._signals_enabled = True

    def disable_signals(self) -> None:
        self._signals_enabled = False

    @contextmanager
    def signals_disabled(self) -> Iterator[None]:
        """Context manager: disable signal updates, restore previous state."""
        prev = self._signals_enabled
        self._signals_enabled = False
        try:
            yield
        finally:
            self._signals_enabled = prev

    # -- layer shapes --------------------------------------------------------

    def layer_shapes(self) -> dict[str, tuple[int, int]]:
        """Shapes metadata for signals (prefer model helper; never weight values)."""
        fn = getattr(self.model, "layer_shapes", None)
        if callable(fn):
            return dict(fn())
        shapes: dict[str, tuple[int, int]] = {}
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Linear):
                shapes[name] = (mod.out_features, mod.in_features)
        if not shapes:
            raise ValueError("model has no Linear layers / layer_shapes()")
        return shapes

    def _activations_for_signals(self) -> dict[str, Tensor]:
        acts = getattr(self.model, "last_activations", None)
        if acts is None:
            raise RuntimeError(
                "model.last_activations is None; forward must populate activation hooks"
            )
        # Detached clones so signal bookkeeping cannot touch the autograd graph.
        return {k: v.detach() for k, v in acts.items()}

    # -- train step ----------------------------------------------------------

    def train_step(self, x: Tensor, y: Tensor) -> dict[str, float]:
        """One instrumented train step.

        Order: forward → encode broadcast → update signals (if enabled) →
        ``loss.backward()`` → ``optimizer.step()``. Signals never enter the loss.
        """
        self.model.train()
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        if y.ndim > 1 and not y.is_floating_point():
            y = y.view(-1)

        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(x)
        loss = self.criterion(logits, y)

        # Broadcast residual packet (detached residuals inside encoder).
        packet = self.encoder.encode(logits, y, loss)
        self._last_packet = packet

        # Local signal updates — read-only observation path.
        if self._signals_enabled and self.signals:
            activations = self._activations_for_signals()
            shapes = self.layer_shapes()
            with torch.no_grad():
                for sig in self.signals:
                    sig.update(activations, shapes, packet, step=self.global_step)

        # Normal backprop only — signals do not affect grads / optimizer.
        loss.backward()
        self.optimizer.step()

        bs = int(y.size(0))
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct = int((preds == y).sum().item())
        metrics = {
            "loss": float(loss.detach().item()),
            "acc": correct / max(bs, 1),
            "n": float(bs),
            "step": float(self.global_step),
        }
        self._last_loss = metrics["loss"]
        self.global_step += 1
        return metrics

    def train_steps(
        self,
        loader: DataLoader,
        *,
        max_steps: int | None = None,
    ) -> dict[str, float]:
        """Run up to ``max_steps`` instrumented train steps from ``loader``."""
        total_loss = 0.0
        total_correct = 0.0
        total_n = 0.0
        steps = 0
        for batch in loader:
            if max_steps is not None and steps >= max_steps:
                break
            x, y = _unpack_batch(batch)
            m = self.train_step(x, y)
            total_loss += m["loss"] * m["n"]
            total_correct += m["acc"] * m["n"]
            total_n += m["n"]
            steps += 1
        if total_n == 0:
            return {"loss": float("nan"), "acc": float("nan"), "n": 0.0, "steps": 0.0}
        return {
            "loss": total_loss / total_n,
            "acc": total_correct / total_n,
            "n": total_n,
            "steps": float(steps),
        }

    # -- snapshot isolation --------------------------------------------------

    @torch.no_grad()
    def freeze_local_scores(self) -> dict[str, ScoreBundle]:
        """(1) Deep-copy local scores under ``no_grad`` (no GT yet)."""
        return {sig.name: sig.snapshot_scores() for sig in self.signals}

    def run_gt(
        self,
        gt_measures: Sequence[GroundTruthImportance],
        probe_loader: DataLoader,
        *,
        edge_sample_mask: dict[str, Tensor] | None = None,
        criterion: nn.Module | None = None,
        verify_probe_cache: bool = True,
        isolate_training_rng: bool = True,
    ) -> dict[str, ImportanceResult]:
        """(2) Run GT suite with signals disabled.

        Does not call ``LocalSignal.update``. Uses :meth:`signals_disabled` so
        any accidental instrumented hook is inert during probe forwards.

        Snapshot-scoped probe cache
        ---------------------------
        Materializes the probe set once under the current TaskContext version
        and passes the same :class:`~neuronca.importance.probe_cache.ProbeCache`
        to every GT measure. The cache is never reused across ``run_gt`` /
        ``snapshot`` calls. When ``verify_probe_cache`` is True (default),
        re-materializes from ``probe_loader`` after the suite and asserts
        bit-equality with the cache (corruption guard).

        Training RNG isolation (always on by default)
        ---------------------------------------------
        Captures ``torch.get_rng_state()`` (and CUDA RNG when present) before
        GT and restores it after, then asserts equality. Closes the class of
        bugs where GT-internal ``DataLoader`` iterator creation (or any future
        sampling) advances the global RNG and silently reorders subsequent
        shuffled training batches. Same "cheap, always-on production guard"
        posture as signal-trace isolation and probe-cache integrity.

        Pre-patch historical note: frozen B1/B3 primary results under
        ``results/historical/pre_patch_rng_isolation/`` were collected
        *without* this bracket. Under that regime, GT probe iteration count
        (driven by edge-sample size and neuron ablation unit count) advanced
        the global CPU RNG and thus coupled into post-snapshot train shuffle
        order. Those archives remain valid for their fixed config; they are
        not bit-comparable to post-patch runs or to re-runs with a different
        ``edge_ablation_sample``.
        """
        from neuronca.importance.probe_cache import (
            assert_probe_cache_matches_loader,
            materialize_probe_cache,
        )

        rng_before = (
            capture_torch_rng_state() if isolate_training_rng else None
        )
        try:
            crit = criterion if criterion is not None else self.criterion
            device = self.device
            # Fresh cache every call — snapshot-scoped, never run-level.
            probe_cache = materialize_probe_cache(probe_loader, device=device)
            results: dict[str, ImportanceResult] = {}
            with self.signals_disabled():
                for gt in gt_measures:
                    results[gt.name] = gt.compute(
                        self.model,
                        probe_cache,
                        crit,
                        edge_sample_mask=edge_sample_mask,
                    )
            if verify_probe_cache:
                assert_probe_cache_matches_loader(
                    probe_cache, probe_loader, device=device
                )
            return results
        finally:
            if isolate_training_rng and rng_before is not None:
                restore_torch_rng_state(rng_before)
                # Always-on equality guard: restore must have fully succeeded.
                assert_torch_rng_state_equal(
                    rng_before,
                    capture_torch_rng_state(),
                    msg=(
                        "torch RNG state not restored after GT/snapshot "
                        "(training RNG isolation violated)"
                    ),
                )

    def snapshot(
        self,
        gt_measures: Sequence[GroundTruthImportance],
        probe_loader: DataLoader,
        *,
        edge_sample_mask: dict[str, Tensor] | None = None,
        criterion: nn.Module | None = None,
        check_isolation: bool = False,
        verify_probe_cache: bool = True,
        isolate_training_rng: bool = True,
    ) -> SnapshotResult:
        """Full snapshot protocol: freeze → GT (signals off) → resume.

        Args:
            gt_measures: GT modules to run (e.g. ``GradientSaliency``).
            probe_loader: Current-task probe DataLoader (source of truth for
                materializing the snapshot-scoped probe cache).
            edge_sample_mask: Optional fixed mask for sparse GT-2.
            criterion: Loss for GT; defaults to trainer criterion.
            check_isolation: If True, assert traces bitwise-identical across GT.
            verify_probe_cache: If True (default), after GT assert the
                snapshot probe cache still matches a fresh loader pass.
            isolate_training_rng: If True (default), bracket GT with torch
                RNG save/restore + equality assert so GT cannot perturb
                subsequent training shuffle order.

        Returns:
            :class:`SnapshotResult` with frozen scores and GT maps.
        """
        # Capture traces before GT when isolation-checking.
        pre_traces: list[dict[str, Any]] | None = None
        if check_isolation:
            pre_traces = [capture_signal_traces(s) for s in self.signals]

        # (1) freeze/copy local scores under no_grad
        local_scores = self.freeze_local_scores()

        # (2) GT with signals disabled (probe cache + RNG isolation inside run_gt)
        gt_results = self.run_gt(
            gt_measures,
            probe_loader,
            edge_sample_mask=edge_sample_mask,
            criterion=criterion,
            verify_probe_cache=verify_probe_cache,
            isolate_training_rng=isolate_training_rng,
        )

        # Isolation assert: traces unchanged
        if check_isolation and pre_traces is not None:
            for sig, before in zip(self.signals, pre_traces):
                after = capture_signal_traces(sig)
                assert_traces_bitwise_equal(
                    before,
                    after,
                    msg=(
                        f"snapshot isolation violated for signal {sig.name!r}: "
                        "traces changed across GT-only call"
                    ),
                )

        # (3) resume training — signals re-enabled by signals_disabled() exit
        assert self.signals_enabled, "signals must be re-enabled after snapshot"

        return SnapshotResult(
            step=self.global_step,
            local_scores=local_scores,
            gt_results=gt_results,
        )


# ---------------------------------------------------------------------------
# Integration helper (train a few steps → isolation check)
# ---------------------------------------------------------------------------


def run_train_then_snapshot_isolation_check(
    model: nn.Module,
    train_loader: DataLoader,
    probe_loader: DataLoader,
    signals: Sequence[LocalSignal],
    *,
    optimizer: Optimizer | None = None,
    gt_measures: Sequence[GroundTruthImportance] | None = None,
    encoder: BroadcastErrorEncoder | None = None,
    criterion: nn.Module | None = None,
    device: str | torch.device | None = None,
    train_steps: int = 3,
    edge_sample_mask: dict[str, Tensor] | None = None,
    build_optimizer: Callable[..., Optimizer] | None = None,
) -> SnapshotResult:
    """Train a few steps with signals updating, then run isolation snapshot.

    Convenience wiring for tests / smoke (not a full experiment runner).

    1. Instrumented train for ``train_steps`` (signals update).
    2. Capture eligibility (and other) traces.
    3. Snapshot: freeze scores + full GT path with signals disabled.
    4. Assert traces bitwise-identical; return the snapshot result.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        if build_optimizer is not None:
            optimizer = build_optimizer(model)
        else:
            from neuronca.training.optim import build_optimizer as _build_opt

            optimizer = _build_opt(model)

    if gt_measures is None:
        from neuronca.importance.gradient_saliency import GradientSaliency

        gt_measures = [GradientSaliency()]

    trainer = InstrumentedTrainer(
        model,
        optimizer,
        signals,
        encoder=encoder,
        criterion=criterion,
        device=device,
    )

    # Train with signals updating.
    trainer.train_steps(train_loader, max_steps=train_steps)
    if trainer.global_step < 1:
        raise RuntimeError("train_loader yielded no batches; cannot check isolation")

    # Explicit pre-capture of eligibility edge scores (test contract T).
    pre = [capture_signal_traces(s) for s in signals]

    result = trainer.snapshot(
        gt_measures,
        probe_loader,
        edge_sample_mask=edge_sample_mask,
        criterion=criterion,
        check_isolation=True,
    )

    # Redundant explicit bitwise check on edge scores + any internal traces present.
    for sig, before in zip(signals, pre):
        after = capture_signal_traces(sig)
        preferred = (
            "edge_scores",
            "neuron_scores",
            "_T_edge",
            "_edge",
            "_neuron",
            "_m_a_edge",
            "_m_ae_edge",
            "_m_e",
        )
        keys = tuple(k for k in preferred if k in before and k in after)
        assert_traces_bitwise_equal(
            before,
            after,
            keys=keys,
            msg=f"eligibility/trace bitwise identity failed for {sig.name!r}",
        )

    return result


def update_signals_from_batch(
    signals: Iterable[LocalSignal],
    activations: Mapping[str, Tensor],
    layer_shapes: Mapping[str, tuple[int, int]],
    packet: BroadcastPacket,
    step: int,
) -> None:
    """Update all candidates under ``no_grad`` (standalone hook helper)."""
    shapes = dict(layer_shapes)
    acts = {k: v.detach() for k, v in activations.items()}
    with torch.no_grad():
        for sig in signals:
            sig.update(acts, shapes, packet, step=step)
