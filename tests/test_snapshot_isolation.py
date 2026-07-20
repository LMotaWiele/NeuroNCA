"""Snapshot isolation: local signal traces must not change across GT-only calls.

Design contract (Evaluation Schedule / Implementation Notes):
- After train steps, capture T = eligibility edge scores / internal traces.
- Run full GT compute path with signals disabled.
- Assert T bitwise-equal after.
- Optionally assert GT does not call ``LocalSignal.update`` (mock).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neuronca.broadcast import BroadcastErrorEncoder
from neuronca.importance import AblationImportance, GradientSaliency
from neuronca.io.seed import set_seed
from neuronca.models.mlp import MLPReasoningNet
from neuronca.signals import (
    ActivationMagnitude,
    GEVBVectorEligibility,
    HebbianResidualEnergyCorr,
    ThreeFactorScalarEligibility,
)
from neuronca.signals.base import ScoreBundle
from neuronca.training.callbacks import (
    InstrumentedTrainer,
    assert_traces_bitwise_equal,
    capture_signal_traces,
    run_train_then_snapshot_isolation_check,
    traces_bitwise_equal,
)
from neuronca.training.optim import build_optimizer


# Tiny net for fast CPU tests
TINY_IN = 16
TINY_HIDDEN = (8, 6, 4)
TINY_C = 4


def _tiny_model(seed: int = 0) -> MLPReasoningNet:
    torch.manual_seed(seed)
    return MLPReasoningNet(
        input_dim=TINY_IN, hidden=TINY_HIDDEN, num_classes=TINY_C
    )


def _toy_loader(
    n: int = 32,
    batch_size: int = 8,
    seed: int = 0,
    input_dim: int = TINY_IN,
    num_classes: int = TINY_C,
) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, input_dim, generator=g)
    y = torch.randint(0, num_classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _all_signals() -> list:
    return [
        ActivationMagnitude(decay=0.99),
        HebbianResidualEnergyCorr(decay=0.99),
        ThreeFactorScalarEligibility(decay=0.99),
        GEVBVectorEligibility(
            decay=0.99,
            experiment_seed=0,
            gevb_feedback_seed=0,
            num_classes=TINY_C,
        ),
    ]


# ---------------------------------------------------------------------------
# Trace capture helpers
# ---------------------------------------------------------------------------


def test_capture_and_compare_traces_equal_on_clone():
    sig = ThreeFactorScalarEligibility(decay=0.99)
    model = _tiny_model()
    x = torch.randn(4, TINY_IN)
    y = torch.randint(0, TINY_C, (4,))
    logits = model(x)
    packet = BroadcastErrorEncoder().encode(logits, y, nn.CrossEntropyLoss()(logits, y))
    sig.update(model.last_activations, model.layer_shapes(), packet, step=0)  # type: ignore[arg-type]

    a = capture_signal_traces(sig)
    b = capture_signal_traces(sig)
    assert traces_bitwise_equal(a, b)
    assert_traces_bitwise_equal(a, b)


def test_traces_detect_mutation():
    sig = ThreeFactorScalarEligibility(decay=0.99)
    model = _tiny_model()
    loader = _toy_loader(n=16, batch_size=8)
    trainer = InstrumentedTrainer(
        model, build_optimizer(model), [sig], device="cpu"
    )
    trainer.train_steps(loader, max_steps=2)
    before = capture_signal_traces(sig)
    # Mutate internal T
    with torch.no_grad():
        for t in sig._T_edge.values():
            t.add_(1.0)
    after = capture_signal_traces(sig)
    assert not traces_bitwise_equal(before, after, keys=("_T_edge",))


# ---------------------------------------------------------------------------
# Instrumented train step: signals update, loss independent of signals
# ---------------------------------------------------------------------------


def test_train_step_updates_signals_and_steps_optimizer():
    set_seed(0)
    model = _tiny_model()
    elig = ThreeFactorScalarEligibility(decay=0.99)
    opt = build_optimizer(model, lr=0.05)
    trainer = InstrumentedTrainer(model, opt, [elig], device="cpu")

    w_before = model.fc1.weight.detach().clone()
    loader = _toy_loader(n=16, batch_size=8, seed=1)
    metrics = trainer.train_steps(loader, max_steps=2)

    assert metrics["steps"] == 2.0
    assert trainer.global_step == 2
    assert elig.edge_scores()  # non-empty after updates
    # Scores should be non-zero somewhere after a few steps of noise data.
    total_abs = sum(v.abs().sum().item() for v in elig.edge_scores().values())
    assert total_abs > 0.0
    # Weights moved via backprop.
    assert not torch.equal(w_before, model.fc1.weight.detach())


def test_signals_disabled_skips_update():
    set_seed(0)
    model = _tiny_model()
    elig = ThreeFactorScalarEligibility(decay=0.99)
    trainer = InstrumentedTrainer(
        model, build_optimizer(model), [elig], device="cpu"
    )
    loader = _toy_loader(n=16, batch_size=8)

    with trainer.signals_disabled():
        assert trainer.signals_enabled is False
        trainer.train_steps(loader, max_steps=2)

    # Never allocated / empty scores
    assert elig.edge_scores() == {} or all(
        v.abs().sum() == 0 for v in elig.edge_scores().values()
    )
    # Flag restored
    assert trainer.signals_enabled is True


def test_train_step_order_encodes_broadcast_before_backprop():
    """Encoder step increments once per train step; packet stored."""
    model = _tiny_model()
    encoder = BroadcastErrorEncoder()
    trainer = InstrumentedTrainer(
        model,
        build_optimizer(model),
        [ActivationMagnitude()],
        encoder=encoder,
        device="cpu",
    )
    assert encoder.step == 0
    x = torch.randn(4, TINY_IN)
    y = torch.randint(0, TINY_C, (4,))
    trainer.train_step(x, y)
    assert encoder.step == 1
    assert trainer._last_packet is not None
    assert trainer._last_packet.e_s_per_sample.shape == (4,)


# ---------------------------------------------------------------------------
# Core isolation: train → capture T → GT disabled → bitwise equal
# ---------------------------------------------------------------------------


def test_eligibility_traces_bitwise_equal_across_gt():
    """After training steps, capture T; GT with signals off; T bitwise equal."""
    set_seed(0)
    model = _tiny_model()
    elig = ThreeFactorScalarEligibility(decay=0.99)
    gevb = GEVBVectorEligibility(
        decay=0.99, experiment_seed=0, num_classes=TINY_C
    )
    hebb = HebbianResidualEnergyCorr(decay=0.99)
    act = ActivationMagnitude(decay=0.99)
    signals = [act, hebb, elig, gevb]

    train_loader = _toy_loader(n=40, batch_size=8, seed=0)
    probe_loader = _toy_loader(n=24, batch_size=8, seed=1)
    trainer = InstrumentedTrainer(
        model, build_optimizer(model), signals, device="cpu"
    )

    # Train with signals updating
    trainer.train_steps(train_loader, max_steps=4)
    assert trainer.global_step == 4

    # Capture T = eligibility edge scores + internal traces
    T_before = {s.name: capture_signal_traces(s) for s in signals}
    elig_edge_before = {
        k: v.clone() for k, v in elig.edge_scores().items()
    }
    elig_T_before = {
        k: v.detach().clone() for k, v in elig._T_edge.items()
    }

    # Full GT path with signals disabled
    gt_measures = [
        GradientSaliency(),
        AblationImportance(
            edge_ablation_sample=4,
            seed=0,
            neuron_ablation="all_hidden",
            hidden_layers=("fc1", "fc2"),
        ),
    ]
    result = trainer.snapshot(
        gt_measures,
        probe_loader,
        check_isolation=True,
    )

    # Signals re-enabled after snapshot
    assert trainer.signals_enabled is True

    # Bitwise identity of eligibility edge scores
    elig_edge_after = elig.edge_scores()
    for layer in elig_edge_before:
        assert torch.equal(elig_edge_before[layer], elig_edge_after[layer]), layer

    # Internal T traces bitwise equal
    for layer in elig_T_before:
        assert torch.equal(elig_T_before[layer], elig._T_edge[layer]), layer

    # All candidates
    for s in signals:
        assert_traces_bitwise_equal(
            T_before[s.name],
            capture_signal_traces(s),
            msg=f"traces changed for {s.name}",
        )

    # Snapshot returned frozen scores + GT
    assert set(result.local_scores.keys()) == {s.name for s in signals}
    assert "grad_abs" in result.gt_results
    assert "ablation_relu" in result.gt_results
    for name, bundle in result.local_scores.items():
        assert isinstance(bundle, ScoreBundle)
        assert bundle.name == name
        assert bundle.edge  # non-empty after training


def test_gt_does_not_call_signal_update_mock():
    """Optional: mock LocalSignal.update and assert it is never called during GT."""
    set_seed(0)
    model = _tiny_model()
    real = ThreeFactorScalarEligibility(decay=0.99)
    train_loader = _toy_loader(n=16, batch_size=8)
    probe_loader = _toy_loader(n=16, batch_size=8, seed=2)

    trainer = InstrumentedTrainer(
        model, build_optimizer(model), [real], device="cpu"
    )
    trainer.train_steps(train_loader, max_steps=2)

    # Wrap update with a mock after training so only GT path is observed.
    original_update = real.update
    mock_update = MagicMock(side_effect=original_update)
    real.update = mock_update  # type: ignore[method-assign]

    mock_update.reset_mock()
    trainer.snapshot([GradientSaliency()], probe_loader, check_isolation=True)
    mock_update.assert_not_called()

    # GT alone also must not invoke update
    mock_update.reset_mock()
    with trainer.signals_disabled():
        GradientSaliency().compute(model, probe_loader, nn.CrossEntropyLoss())
    mock_update.assert_not_called()


def test_run_train_then_snapshot_isolation_check_helper():
    """Helper: train a few steps with signals updating, then isolation check."""
    set_seed(1)
    model = _tiny_model(seed=1)
    signals = _all_signals()
    train_loader = _toy_loader(n=32, batch_size=8, seed=3)
    probe_loader = _toy_loader(n=16, batch_size=8, seed=4)

    result = run_train_then_snapshot_isolation_check(
        model,
        train_loader,
        probe_loader,
        signals,
        train_steps=3,
        device="cpu",
        gt_measures=[GradientSaliency()],
    )

    assert result.step == 3
    assert "grad_abs" in result.gt_results
    assert "eligibility_residual_energy" in result.local_scores
    # Frozen scores match live (no further train steps)
    live = signals[2]  # ThreeFactor
    frozen = result.local_scores["eligibility_residual_energy"]
    for layer in frozen.edge:
        assert torch.equal(frozen.edge[layer], live.edge_scores()[layer])


def test_snapshot_isolation_with_disabled_flag_during_gt_only_call():
    """Direct GT-only path: disable → capture → GT → assert equal."""
    set_seed(0)
    model = _tiny_model()
    elig = ThreeFactorScalarEligibility(decay=0.99)
    trainer = InstrumentedTrainer(
        model, build_optimizer(model), [elig], device="cpu"
    )
    trainer.train_steps(_toy_loader(n=24, batch_size=8), max_steps=3)

    T = capture_signal_traces(elig)
    with trainer.signals_disabled():
        # Accidental train_step must not update traces
        trainer.train_step(torch.randn(4, TINY_IN), torch.randint(0, TINY_C, (4,)))
        GradientSaliency().compute(
            model, _toy_loader(n=16, batch_size=8, seed=9), nn.CrossEntropyLoss()
        )
    assert_traces_bitwise_equal(T, capture_signal_traces(elig))
    # Optimizer still stepped under disabled signals (weights may change) —
    # but traces must not. That is the isolation property under test.


def test_freeze_local_scores_is_deepcopy():
    set_seed(0)
    model = _tiny_model()
    elig = ThreeFactorScalarEligibility(decay=0.99)
    trainer = InstrumentedTrainer(
        model, build_optimizer(model), [elig], device="cpu"
    )
    trainer.train_steps(_toy_loader(n=16, batch_size=8), max_steps=2)

    frozen = trainer.freeze_local_scores()
    bundle = frozen[elig.name]
    # Mutate live scores via another train step
    trainer.train_steps(_toy_loader(n=16, batch_size=8, seed=5), max_steps=1)
    live = elig.edge_scores()
    # Frozen must be independent
    changed = False
    for layer in bundle.edge:
        if not torch.equal(bundle.edge[layer], live[layer]):
            changed = True
            break
    assert changed, "expected live scores to move after further training"
