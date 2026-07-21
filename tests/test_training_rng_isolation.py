"""Training RNG isolation: GT/snapshot must not advance the train-stream RNG."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neuronca.importance.gradient_saliency import GradientSaliency
from neuronca.io.seed import set_seed
from neuronca.models.mlp import MLPReasoningNet
from neuronca.training.callbacks import (
    InstrumentedTrainer,
    TrainingRNGIsolationError,
    assert_torch_rng_state_equal,
    capture_torch_rng_state,
    restore_torch_rng_state,
)
from neuronca.training.optim import build_optimizer


def _loader(n: int = 32, batch_size: int = 8, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 16, generator=g)
    y = torch.randint(0, 4, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def test_capture_restore_roundtrip():
    set_seed(0)
    state = capture_torch_rng_state()
    a = torch.rand(3)
    _ = torch.rand(3)  # advance past a
    restore_torch_rng_state(state)
    b = torch.rand(3)
    assert torch.equal(a, b)


def test_assert_detects_cpu_rng_drift():
    set_seed(1)
    before = capture_torch_rng_state()
    _ = torch.rand(1)
    after = capture_torch_rng_state()
    try:
        assert_torch_rng_state_equal(before, after)
        raised = False
    except TrainingRNGIsolationError:
        raised = True
    assert raised


def test_run_gt_restores_cpu_rng_despite_probe_loader_iters():
    """Many probe DataLoader passes must not leak into the global CPU RNG."""
    set_seed(0)
    model = MLPReasoningNet(input_dim=16, hidden=(8, 6, 4), num_classes=4)
    opt = build_optimizer(model)
    trainer = InstrumentedTrainer(model, opt, signals=[], device="cpu")
    probe = _loader(n=24, batch_size=6, seed=2)

    # Advance RNG a known amount, capture, run GT (which iterates probe many times
    # if uncached — with cache, still creates at least one iterator), restore.
    _ = torch.rand(5)
    before = capture_torch_rng_state()
    trainer.run_gt([GradientSaliency()], probe, verify_probe_cache=True)
    after = capture_torch_rng_state()
    assert_torch_rng_state_equal(before, after)

    # Next random draw matches a fresh restore from the pre-GT state.
    restore_torch_rng_state(before)
    expected = torch.rand(4)
    restore_torch_rng_state(before)
    trainer.run_gt([GradientSaliency()], probe)
    got = torch.rand(4)
    assert torch.equal(expected, got)


def test_snapshot_rng_isolation_default_on():
    set_seed(7)
    model = MLPReasoningNet(input_dim=16, hidden=(8, 6, 4), num_classes=4)
    trainer = InstrumentedTrainer(
        model, build_optimizer(model), signals=[], device="cpu"
    )
    probe = _loader(seed=3)
    train = _loader(seed=4)
    trainer.train_steps(train, max_steps=2)

    _ = torch.rand(2)
    before = capture_torch_rng_state()
    trainer.snapshot([GradientSaliency()], probe, check_isolation=False)
    after = capture_torch_rng_state()
    assert_torch_rng_state_equal(before, after)


def test_isolate_training_rng_false_allows_leak():
    """Opt-out path: without isolation, DataLoader iters advance global RNG."""
    set_seed(0)
    model = MLPReasoningNet(input_dim=16, hidden=(8, 6, 4), num_classes=4)
    trainer = InstrumentedTrainer(
        model, build_optimizer(model), signals=[], device="cpu"
    )
    probe = _loader(n=16, batch_size=4, seed=9)
    before = capture_torch_rng_state()
    trainer.run_gt(
        [GradientSaliency()],
        probe,
        verify_probe_cache=False,
        isolate_training_rng=False,
    )
    after = capture_torch_rng_state()
    # With verify_probe_cache=False still materializes once → at least one iter.
    # Isolation off → state should differ.
    assert not torch.equal(before["cpu"], after["cpu"])
