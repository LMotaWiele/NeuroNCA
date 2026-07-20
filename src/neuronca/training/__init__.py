"""Backprop training loop and snapshot isolation callbacks."""

from neuronca.training.callbacks import (
    InstrumentedTrainer,
    SnapshotResult,
    assert_traces_bitwise_equal,
    capture_signal_traces,
    run_train_then_snapshot_isolation_check,
    traces_bitwise_equal,
    update_signals_from_batch,
)
from neuronca.training.loop import evaluate, resolve_device, train_epoch, train_model
from neuronca.training.optim import build_optimizer

__all__ = [
    "InstrumentedTrainer",
    "SnapshotResult",
    "assert_traces_bitwise_equal",
    "build_optimizer",
    "capture_signal_traces",
    "evaluate",
    "resolve_device",
    "run_train_then_snapshot_isolation_check",
    "traces_bitwise_equal",
    "train_epoch",
    "train_model",
    "update_signals_from_batch",
]
