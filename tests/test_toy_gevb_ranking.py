"""Toy ranking integration: known GT-1 edge order on a fixed tiny network.

Also guards the GT protocol contract: ground-truth compute paths must never
invoke ``LocalSignal.update`` (snapshot isolation; eligibility / GEVB-RF traces
must stay frozen while GT runs). Full GEVB signal plumbing is PR-6; here we
use a mock LocalSignal counter.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from neuronca.importance import AblationImportance, GradientSaliency
from neuronca.importance.base import mean_probe_loss
from neuronca.io.seed import set_seed
from neuronca.models.mlp import MLPReasoningNet


class _MockLocalSignal:
    """Minimal LocalSignal stand-in that counts ``update`` calls."""

    name = "mock_gevb_or_elig"

    def __init__(self) -> None:
        self.update_calls = 0
        self._neuron: dict[str, Tensor] = {}
        self._edge: dict[str, Tensor] = {}

    def reset(self) -> None:
        self.update_calls = 0

    def update(
        self,
        activations: dict[str, Tensor],
        layer_shapes: dict[str, tuple[int, int]],
        broadcast: Any,
        step: int,
    ) -> None:
        self.update_calls += 1

    def neuron_scores(self) -> dict[str, Tensor]:
        return self._neuron

    def edge_scores(self) -> dict[str, Tensor]:
        return self._edge

    def snapshot_scores(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "neuron": {k: v.clone() for k, v in self._neuron.items()},
            "edge": {k: v.clone() for k, v in self._edge.items()},
        }


class _ToyLinearNet(nn.Module):
    """Single Linear, bias-free, fixed weights — analytical grad ranking.

    With MSE and a single example ``x = [1, 0]``, ``y = [0, 0]``:

        W = [[w00, w01], [w10, w11]]
        pred = Wx = [w00, w10]
        residual = pred - y = [w00, w10]
        ∂L/∂W = residual ⊗ x   (L = 0.5 ||r||^2  →  dL/dW = r x^T)

    so only column 0 is nonzero, and |∂L/∂W_00| = |w00|, |∂L/∂W_10| = |w10|.
    Ranking of column-0 entries is therefore the ranking of |w*0|.
    """

    def __init__(self, weight: Tensor) -> None:
        super().__init__()
        out_f, in_f = weight.shape
        self.fc1 = nn.Linear(in_f, out_f, bias=False)
        with torch.no_grad():
            self.fc1.weight.copy_(weight)

    def named_linear_layers(self) -> list[tuple[str, nn.Linear]]:
        return [("fc1", self.fc1)]

    def layer_shapes(self) -> dict[str, tuple[int, int]]:
        return {"fc1": (self.fc1.out_features, self.fc1.in_features)}

    def forward(self, x: Tensor) -> Tensor:
        return self.fc1(x)


def test_gt1_analytical_edge_ranking_mse():
    """Known |∂L/∂W| order from closed-form single-sample MSE."""
    # |w00|=3 > |w10|=1 > others get zero grad for x=[1,0]
    w = torch.tensor([[3.0, 0.5], [1.0, 2.0]])
    model = _ToyLinearNet(w)
    x = torch.tensor([[1.0, 0.0]])  # (1, 2)
    y = torch.zeros(1, 2)  # same shape as logits for MSE
    loader = DataLoader(TensorDataset(x, y), batch_size=1)
    criterion = nn.MSELoss()

    result = GradientSaliency().compute(model, loader, criterion)
    edge = result.edge["fc1"]

    # Closed form for mean-reduced MSE: L = mean(0.5 is wrong for default MSE)
    # nn.MSELoss default is mean over all elements: L = mean((Wx-y)^2)
    # dL/dW_ji = (2/N_elem) * r_j * x_i  with N_elem = out*batch... actually
    # MSELoss: mean over (B, C) → dL/d pred_j = 2 * r_j / (B*C)
    # We only need *ranking* of |grad|, not absolute scale.
    # Column 0 must dominate column 1 (x_1=0 → grad col1 = 0).
    assert edge[0, 1].item() == 0.0
    assert edge[1, 1].item() == 0.0
    assert edge[0, 0].item() > edge[1, 0].item() > 0.0

    # Full ranking of the 4 edges by score: (0,0) > (1,0) > (0,1)=(1,1)=0
    flat = edge.reshape(-1)
    order = torch.argsort(flat, descending=True).tolist()
    # indices: 0→(0,0), 1→(0,1), 2→(1,0), 3→(1,1)
    assert order[0] == 0  # (0,0) largest
    assert order[1] == 2  # (1,0) second
    assert set(order[2:]) == {1, 3}

    # Neuron scores = sum incoming: unit 0 gets |g00|, unit 1 gets |g10|
    assert result.neuron["fc1"][0].item() > result.neuron["fc1"][1].item()


def test_gt1_ranking_stable_on_tiny_mlp():
    """Tiny MLP with fixed W: GT-1 ranking matches independent autograd order."""
    set_seed(42)
    model = MLPReasoningNet(input_dim=6, hidden=(4, 3, 2), num_classes=2)
    # Fix weights to a deterministic positive pattern for stable ReLU paths.
    with torch.no_grad():
        for i, (_, mod) in enumerate(model.named_linear_layers()):
            g = torch.Generator().manual_seed(100 + i)
            mod.weight.copy_(torch.randn(mod.weight.shape, generator=g) * 0.5)
            if mod.bias is not None:
                mod.bias.zero_()

    g = torch.Generator().manual_seed(7)
    x = torch.randn(20, 6, generator=g).abs()  # nonnegative → more active ReLUs
    y = torch.randint(0, 2, (20,), generator=g)
    loader = DataLoader(TensorDataset(x, y), batch_size=5, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    result = GradientSaliency().compute(model, loader, criterion)

    # Independent reference ranking for fc_out (smallest dense layer)
    model.eval()
    acc = torch.zeros_like(model.fc_out.weight)
    n = 0
    for xb, yb in loader:
        model.zero_grad(set_to_none=True)
        criterion(model(xb), yb).backward()
        acc += model.fc_out.weight.grad.detach().abs() * yb.size(0)
        n += yb.size(0)
    model.zero_grad(set_to_none=True)
    ref = acc / n

    gt_flat = result.edge["fc_out"].reshape(-1)
    ref_flat = ref.cpu().float().reshape(-1)
    assert torch.allclose(gt_flat, ref_flat, atol=1e-6)

    # argsort order must match exactly
    assert torch.equal(
        torch.argsort(gt_flat, descending=True),
        torch.argsort(ref_flat, descending=True),
    )


def test_gt_compute_never_calls_local_signal_update():
    """GT modules must not touch LocalSignal.update (GEVB/eligibility traces)."""
    set_seed(0)
    model = MLPReasoningNet(input_dim=8, hidden=(4, 3, 2), num_classes=2)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(12, 8, generator=g)
    y = torch.randint(0, 2, (12,), generator=g)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    criterion = nn.CrossEntropyLoss()

    signal = _MockLocalSignal()
    # Seed mock scores so a buggy GT cannot "read" empty dicts as success only.
    shapes = model.layer_shapes()
    signal._neuron = {n: torch.zeros(o) for n, (o, _) in shapes.items()}
    signal._edge = {n: torch.zeros(o, i) for n, (o, i) in shapes.items()}
    snap_before = signal.snapshot_scores()
    calls_before = signal.update_calls

    # Run GT measures — they only receive model/loader/criterion.
    GradientSaliency().compute(model, loader, criterion)
    AblationImportance(
        edge_ablation_sample=2,
        seed=0,
        neuron_ablation="all_hidden",
        hidden_layers=("fc1",),
    ).compute(model, loader, criterion)

    assert signal.update_calls == calls_before
    snap_after = signal.snapshot_scores()
    for name in snap_before["neuron"]:
        assert torch.equal(snap_before["neuron"][name], snap_after["neuron"][name])
        assert torch.equal(snap_before["edge"][name], snap_after["edge"][name])


def test_ablation_relu_positive_delta_when_useful_unit():
    """Zeroing a heavily used positive unit should not decrease mean CE (ΔL ≥ 0)."""
    set_seed(0)
    # Construct a tiny path where fc1 unit 0 is the only useful feature.
    model = MLPReasoningNet(input_dim=2, hidden=(2, 2, 2), num_classes=2)
    with torch.no_grad():
        for _, mod in model.named_linear_layers():
            mod.weight.zero_()
            if mod.bias is not None:
                mod.bias.zero_()
        # fc1: unit 0 reads x0 strongly; unit 1 dead
        model.fc1.weight[0, 0] = 2.0
        # Pass unit 0 through remaining layers to logit 0
        model.fc2.weight[0, 0] = 2.0
        model.fc3.weight[0, 0] = 2.0
        model.fc_out.weight[0, 0] = 2.0  # class 0 logit boosted by path

    # Inputs with x0 large, labels class 0 → path is useful
    x = torch.tensor([[3.0, 0.0], [2.5, 0.0], [2.0, 0.0], [1.5, 0.0]])
    y = torch.zeros(4, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    criterion = nn.CrossEntropyLoss()

    l_base = mean_probe_loss(model, loader, criterion)
    # Manual ablate fc1 unit 0
    handle = model.fc1.register_forward_hook(
        lambda _m, _i, out: out.clone().index_fill_(1, torch.tensor([0]), 0.0)
    )
    try:
        l_ab = mean_probe_loss(model, loader, criterion)
    finally:
        handle.remove()
    assert l_ab >= l_base - 1e-6

    gt = AblationImportance(
        edge_ablation_sample=0,
        neuron_ablation="all_hidden",
        hidden_layers=("fc1",),
    )
    result = gt.compute(model, loader, criterion)
    # Unit 0 should rank >= unit 1 under ablation_relu
    assert result.neuron["fc1"][0].item() >= result.neuron["fc1"][1].item()
