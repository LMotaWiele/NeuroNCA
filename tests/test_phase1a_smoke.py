"""Smoke tests for Phase 1a plastic training (tiny net, no full MNIST epochs)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from neuronca.models.dynamic_mlp import DynamicMLPReasoningNet
from neuronca.plasticity.diagnostics import length_histogram, sparsity_report
from neuronca.plasticity.orchestrator import PlasticityOrchestrator
from neuronca.plasticity.rules import PlasticityConfig
from neuronca.training.optim import build_optimizer
from neuronca.training.plastic_loop import train_plastic_model


def test_plastic_train_two_epochs_synthetic():
    torch.manual_seed(0)
    net = DynamicMLPReasoningNet(
        hidden_init=(8, 6, 4),
        hidden_max=(16, 12, 8),
        coord_seed=0,
    )
    opt = build_optimizer(net, "sgd", lr=0.05, momentum=0.0)
    cfg = PlasticityConfig(
        radius=0.5,
        target_rate=0.05,
        tau=0.01,
        threshold_kind="fixed",
        prune_edge_fraction=0.02,
        max_new_edges_per_layer=16,
        max_new_neurons_per_layer=2,
    )
    orch = PlasticityOrchestrator(cfg)
    x = torch.randn(64, 784)
    y = torch.randint(0, 10, (64,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
    out = train_plastic_model(
        net,
        loader,
        opt,
        orch,
        epochs=2,
        device="cpu",
        max_steps_per_epoch=4,
    )
    assert len(out["structure"]) == 2
    assert len(out["train"]) == 2
    assert all(torch.isfinite(torch.tensor(r["loss"])) for r in out["train"])
    sp = sparsity_report(net)
    assert "fc1" in sp
    hist = length_histogram(net)
    assert "init_dense" in hist and "incentive_grown" in hist


def test_plastic_train_sliding_variant():
    torch.manual_seed(1)
    net = DynamicMLPReasoningNet(
        hidden_init=(6, 4, 3),
        hidden_max=(12, 8, 6),
        coord_seed=1,
    )
    opt = build_optimizer(net, "sgd", lr=0.05)
    cfg = PlasticityConfig(
        radius=0.5,
        target_rate=0.05,
        tau=0.01,
        threshold_kind="sliding",
        mu=0.99,
        prune_edge_fraction=0.02,
        max_new_edges_per_layer=8,
        max_new_neurons_per_layer=1,
    )
    orch = PlasticityOrchestrator(cfg)
    x = torch.randn(32, 784)
    y = torch.randint(0, 10, (32,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    out = train_plastic_model(net, loader, opt, orch, epochs=1, max_steps_per_epoch=3)
    assert out["structure"][0]["step"] == 1
