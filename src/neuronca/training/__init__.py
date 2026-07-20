"""Backprop training loop and snapshot isolation callbacks."""

from neuronca.training.loop import evaluate, resolve_device, train_epoch, train_model
from neuronca.training.optim import build_optimizer

__all__ = [
    "build_optimizer",
    "evaluate",
    "resolve_device",
    "train_epoch",
    "train_model",
]
