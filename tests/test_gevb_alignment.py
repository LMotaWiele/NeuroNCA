"""GEVB FA alignment diagnostics."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neuronca.broadcast import BroadcastErrorEncoder
from neuronca.io.seed import set_seed
from neuronca.models.mlp import MLPReasoningNet
from neuronca.signals.gevb_alignment import (
    compute_gevb_alignment,
    subspace_principal_angle_deg,
)
from neuronca.signals.gevb_eligibility import GEVBVectorEligibility
from neuronca.training.callbacks import InstrumentedTrainer
from neuronca.training.optim import build_optimizer


def test_subspace_principal_angle_identical_top_sv_and_B():
    # W with dominant left direction e1; B along e1 → ~0°
    W = torch.zeros(3, 2)
    W[0, 0] = 5.0
    W[1, 1] = 0.1  # small secondary
    B = torch.tensor([[1.0], [0.0], [0.0]])
    ang = subspace_principal_angle_deg(W, B)
    assert ang < 5.0


def test_subspace_principal_angle_orthogonal_to_top_sv():
    # W dominated by e1; B along e2 → near 90°
    W = torch.zeros(3, 2)
    W[0, 0] = 5.0
    W[0, 1] = 0.01
    B = torch.tensor([[0.0], [1.0], [0.0]])
    ang = subspace_principal_angle_deg(W, B)
    assert 80.0 < ang <= 90.0


def test_compute_gevb_alignment_returns_layers_and_finite_angles():
    set_seed(0)
    model = MLPReasoningNet(input_dim=16, hidden=(8, 6, 4), num_classes=4)
    shapes = model.layer_shapes()
    gevb = GEVBVectorEligibility(
        decay=0.99,
        experiment_seed=0,
        num_classes=4,
        layer_shapes=shapes,
    )
    g = torch.Generator().manual_seed(0)
    x = torch.randn(32, 16, generator=g)
    y = torch.randint(0, 4, (32,), generator=g)
    loader = DataLoader(TensorDataset(x, y), batch_size=8)

    # Warm signals / B
    trainer = InstrumentedTrainer(
        model, build_optimizer(model), [gevb], device="cpu"
    )
    trainer.train_steps(loader, max_steps=3)

    result = compute_gevb_alignment(
        model,
        gevb,
        loader,
        criterion=nn.CrossEntropyLoss(),
        encoder=BroadcastErrorEncoder(),
        device="cpu",
        max_batches=2,
    )
    assert set(result.keys()) == set(shapes.keys())
    for layer, metrics in result.items():
        assert "fa_signal_angle_deg" in metrics
        assert "subspace_principal_angle_deg" in metrics
        fa = metrics["fa_signal_angle_deg"]
        sub = metrics["subspace_principal_angle_deg"]
        # FA signal angle is geometric [0, 180]; subspace principal angle [0, 90]
        if math.isfinite(fa):
            assert 0.0 <= fa <= 180.0
        if math.isfinite(sub):
            assert 0.0 <= sub <= 90.0


def test_alignment_does_not_mutate_eligibility_traces():
    set_seed(1)
    model = MLPReasoningNet(input_dim=16, hidden=(8, 6, 4), num_classes=4)
    gevb = GEVBVectorEligibility(
        decay=0.99,
        experiment_seed=1,
        num_classes=4,
        layer_shapes=model.layer_shapes(),
    )
    loader = DataLoader(
        TensorDataset(torch.randn(24, 16), torch.randint(0, 4, (24,))),
        batch_size=8,
    )
    trainer = InstrumentedTrainer(
        model, build_optimizer(model), [gevb], device="cpu"
    )
    trainer.train_steps(loader, max_steps=2)
    before = {k: v.detach().clone() for k, v in gevb._T_edge.items()}

    compute_gevb_alignment(
        model, gevb, loader, device="cpu", max_batches=1
    )

    for k, v in before.items():
        assert torch.equal(v, gevb._T_edge[k])
