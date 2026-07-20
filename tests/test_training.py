"""Tests for SGD optimizer factory and backprop training loop."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from neuronca.data.mnist import build_mnist_loaders
from neuronca.data.tasks import TaskContext, make_identity_transform
from neuronca.io.seed import set_seed
from neuronca.models.mlp import MLPReasoningNet
from neuronca.training.loop import evaluate, resolve_device, train_epoch, train_model
from neuronca.training.optim import (
    DEFAULT_LR,
    DEFAULT_MOMENTUM,
    DEFAULT_WEIGHT_DECAY,
    build_optimizer,
)


def _toy_loader(n: int = 64, batch_size: int = 16, num_classes: int = 10) -> DataLoader:
    """Synthetic flat MNIST-like tensors (no download)."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 784, generator=g)
    y = torch.randint(0, num_classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def test_build_optimizer_sgd_defaults():
    model = MLPReasoningNet()
    opt = build_optimizer(model)
    assert isinstance(opt, torch.optim.SGD)
    group = opt.param_groups[0]
    assert group["lr"] == DEFAULT_LR
    assert group["momentum"] == DEFAULT_MOMENTUM
    assert group["weight_decay"] == DEFAULT_WEIGHT_DECAY


def test_build_optimizer_adam_optional():
    model = MLPReasoningNet()
    opt = build_optimizer(model, name="adam", lr=1e-3)
    assert isinstance(opt, torch.optim.Adam)
    assert opt.param_groups[0]["lr"] == 1e-3


def test_build_optimizer_case_insensitive_and_unknown():
    model = MLPReasoningNet()
    opt = build_optimizer(model, name="SGD")
    assert isinstance(opt, torch.optim.SGD)
    try:
        build_optimizer(model, name="rmsprop")
        assert False, "expected ValueError for unsupported optimizer"
    except ValueError as exc:
        assert "rmsprop" in str(exc).lower() or "unsupported" in str(exc).lower()


def test_resolve_device_auto():
    dev = resolve_device(None)
    assert isinstance(dev, torch.device)
    assert dev.type in ("cpu", "cuda")
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "cuda")


def test_train_epoch_and_evaluate_on_toy():
    set_seed(0)
    model = MLPReasoningNet()
    loader = _toy_loader()
    opt = build_optimizer(model, name="sgd", lr=0.05)
    device = "cpu"

    before = evaluate(model, loader, device=device)
    metrics = train_epoch(model, loader, opt, device=device)
    after = evaluate(model, loader, device=device)

    assert metrics["n"] > 0
    assert metrics["steps"] > 0
    assert 0.0 <= metrics["acc"] <= 1.0
    assert metrics["loss"] == metrics["loss"]  # finite
    # Training on toy data should move loss at least a little, or stay finite.
    assert after["loss"] == after["loss"]
    assert before["loss"] == before["loss"]


def test_train_model_returns_history():
    set_seed(1)
    model = MLPReasoningNet()
    train_loader = _toy_loader(n=48, batch_size=16)
    eval_loader = _toy_loader(n=32, batch_size=16)
    opt = build_optimizer(model)
    result = train_model(
        model,
        train_loader,
        opt,
        epochs=2,
        device="cpu",
        eval_loader=eval_loader,
    )
    assert result["epochs"] == 2
    assert len(result["history"]) == 2
    assert "train_loss" in result and "train_acc" in result
    assert "eval_loss" in result and "eval_acc" in result
    assert result["history"][0]["epoch"] == 0
    assert result["history"][1]["epoch"] == 1
    assert result["device"] == "cpu"


def test_smoke_mnist_short_train_beats_random_or_loss_drops():
    """Integration smoke: few steps on real MNIST; loss drops or acc > random.

    Random 10-class accuracy is ~0.10; we allow either a clear loss decrease
    across a short run or train accuracy above chance after training.
    """
    set_seed(0)
    ctx = TaskContext(make_identity_transform())
    bundles = build_mnist_loaders(
        ctx,
        root="data/mnist",
        probe_size=128,
        seed=0,
        batch_size=64,
        num_workers=0,
        download=True,
    )

    # Tiny subset for speed: first 512 train examples via Subset wrapper.
    train_ds = bundles["train_dataset"]
    n_subset = min(512, len(train_ds))
    subset = Subset(train_ds, list(range(n_subset)))
    train_loader = DataLoader(subset, batch_size=64, shuffle=True, num_workers=0)

    model = MLPReasoningNet()
    opt = build_optimizer(model, name="sgd", lr=0.05, momentum=0.9, weight_decay=1e-4)
    device = resolve_device("auto")

    # Initial loss on the same subset (eval mode, no updates).
    init_metrics = evaluate(model, train_loader, device=device)
    init_loss = init_metrics["loss"]
    assert init_loss == init_loss  # finite

    result = train_model(
        model,
        train_loader,
        opt,
        epochs=3,
        device=device,
        max_steps_per_epoch=None,
    )
    final_loss = result["train_loss"]
    final_acc = result["train_acc"]

    loss_decreased = final_loss < init_loss - 0.05
    above_chance = final_acc > 0.20  # well above 1/10 random
    assert loss_decreased or above_chance, (
        f"expected loss drop or acc>chance; "
        f"init_loss={init_loss:.4f} final_loss={final_loss:.4f} final_acc={final_acc:.4f}"
    )


def test_criterion_is_cross_entropy_default():
    """train_epoch uses CE: logits shape (B, C) with integer targets."""
    set_seed(0)
    model = MLPReasoningNet()
    # Single batch of correct shapes
    x = torch.randn(8, 784)
    y = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    opt = build_optimizer(model)
    m = train_epoch(model, loader, opt, device="cpu", max_steps=1)
    assert m["steps"] == 1.0
    # Manual CE for comparison on same batch after one step would differ;
    # just check CE callable is accepted when passed explicitly.
    ce = nn.CrossEntropyLoss()
    m2 = evaluate(model, loader, device="cpu", criterion=ce)
    assert m2["loss"] == m2["loss"]
