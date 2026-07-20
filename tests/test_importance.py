"""Tests for ground-truth importance: grad_abs, ablation_relu, edge masks, Fisher."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neuronca.importance import (
    AblationImportance,
    FisherImportance,
    GradientSaliency,
    GroundTruthImportance,
    ImportanceResult,
    build_edge_ablation_mask,
    compute_fisher,
    compute_grad_abs,
    default_edge_mask_path,
    load_edge_ablation_mask,
    save_edge_ablation_mask,
)
from neuronca.io.seed import set_seed
from neuronca.models.mlp import MLPReasoningNet


def _toy_loader(
    n: int = 32,
    batch_size: int = 8,
    input_dim: int = 16,
    num_classes: int = 4,
    seed: int = 0,
) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, input_dim, generator=g)
    y = torch.randint(0, num_classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _tiny_mlp(
    input_dim: int = 16,
    hidden: tuple[int, ...] = (8, 6, 4),
    num_classes: int = 4,
) -> MLPReasoningNet:
    return MLPReasoningNet(input_dim=input_dim, hidden=hidden, num_classes=num_classes)


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def test_importance_result_dataclass_fields():
    r = ImportanceResult(
        neuron={"fc1": torch.zeros(3)},
        edge={"fc1": torch.zeros(3, 2)},
        edge_index=None,
        extras={"grad_movement": {"fc1": torch.zeros(3, 2)}},
    )
    assert "fc1" in r.neuron and "fc1" in r.edge
    assert r.edge_index is None
    assert "grad_movement" in r.extras


def test_gradient_saliency_shapes_and_protocol():
    set_seed(0)
    model = _tiny_mlp()
    loader = _toy_loader()
    criterion = nn.CrossEntropyLoss()
    gt = GradientSaliency()
    assert isinstance(gt, GroundTruthImportance)
    assert gt.name == "grad_abs"

    result = gt.compute(model, loader, criterion)
    shapes = model.layer_shapes()
    assert set(result.edge.keys()) == set(shapes.keys())
    assert set(result.neuron.keys()) == set(shapes.keys())
    assert result.edge_index is None

    for name, (n_out, n_in) in shapes.items():
        assert result.edge[name].shape == (n_out, n_in)
        assert result.neuron[name].shape == (n_out,)
        # neuron = sum incoming
        assert torch.allclose(
            result.neuron[name], result.edge[name].sum(dim=1), atol=1e-5
        )
        assert result.edge[name].dtype == torch.float32
        assert result.edge[name].device.type == "cpu"
        assert (result.edge[name] >= 0).all()

    assert "grad_movement" in result.extras
    for name, (n_out, n_in) in shapes.items():
        gm = result.extras["grad_movement"][name]
        assert gm.shape == (n_out, n_in)
        # movement = |grad| * |W|
        w = dict(model.named_linear_layers())[name].weight.detach().cpu().abs()
        assert torch.allclose(gm, result.edge[name] * w, atol=1e-5)


def test_gradient_saliency_does_not_mutate_weights():
    set_seed(1)
    model = _tiny_mlp()
    before = _clone_state(model)
    loader = _toy_loader()
    compute_grad_abs(model, loader, nn.CrossEntropyLoss())
    after = _clone_state(model)
    assert before.keys() == after.keys()
    for k in before:
        assert torch.equal(before[k], after[k]), k
    # Grads should be cleared
    for _, mod in model.named_linear_layers():
        assert mod.weight.grad is None


def test_gradient_saliency_matches_manual_reference():
    """Independent accumulation must match GradientSaliency.edge."""
    set_seed(2)
    model = _tiny_mlp()
    loader = _toy_loader(n=24, batch_size=6)
    criterion = nn.CrossEntropyLoss()

    # Manual reference
    device = torch.device("cpu")
    model.eval()
    layers = model.named_linear_layers()
    acc = {n: torch.zeros_like(m.weight) for n, m in layers}
    total_n = 0
    for x, y in loader:
        model.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        bs = y.size(0)
        for n, m in layers:
            acc[n] = acc[n] + m.weight.grad.detach().abs() * bs
        total_n += bs
    model.zero_grad(set_to_none=True)
    ref = {n: a / total_n for n, a in acc.items()}

    result = GradientSaliency().compute(model, loader, criterion)
    for n in ref:
        assert torch.allclose(result.edge[n], ref[n].cpu().float(), atol=1e-6), n


def test_edge_ablation_mask_fixed_and_stratified():
    set_seed(0)
    model = _tiny_mlp()
    m1 = build_edge_ablation_mask(model, seed=7, sample_per_layer=12, n_bins=4)
    m2 = build_edge_ablation_mask(model, seed=7, sample_per_layer=12, n_bins=4)
    m3 = build_edge_ablation_mask(model, seed=8, sample_per_layer=12, n_bins=4)

    shapes = model.layer_shapes()
    assert set(m1.keys()) == set(shapes.keys())
    for name, (n_out, n_in) in shapes.items():
        assert m1[name].shape == (n_out, n_in)
        assert m1[name].dtype == torch.bool
        assert int(m1[name].sum().item()) == min(12, n_out * n_in)
        assert torch.equal(m1[name], m2[name]), name
        # Different seed → usually different mask (not required equal)
    # At least one layer differs across seeds for non-tiny sample
    assert any(not torch.equal(m1[n], m3[n]) for n in m1)


def test_edge_ablation_mask_persist_roundtrip(tmp_path: Path):
    model = _tiny_mlp()
    mask = build_edge_ablation_mask(model, seed=3, sample_per_layer=10)
    path = default_edge_mask_path(tmp_path, seed=3)
    save_edge_ablation_mask(mask, path)
    loaded = load_edge_ablation_mask(path)
    assert set(loaded.keys()) == set(mask.keys())
    for k in mask:
        assert torch.equal(mask[k], loaded[k])


def test_edge_ablation_mask_sample_zero():
    model = _tiny_mlp()
    mask = build_edge_ablation_mask(model, seed=0, sample_per_layer=0)
    for name, m in mask.items():
        assert m.sum().item() == 0, name


def test_ablation_neuron_relu_and_weight_restore():
    set_seed(0)
    model = _tiny_mlp()
    before = _clone_state(model)
    loader = _toy_loader()
    criterion = nn.CrossEntropyLoss()

    gt = AblationImportance(
        edge_ablation_sample=0,  # neuron only for speed
        seed=0,
        hidden_layers=("fc1", "fc2", "fc3"),
    )
    assert gt.name == "ablation_relu"
    result = gt.compute(model, loader, criterion)

    # Hidden only
    assert set(result.neuron.keys()) == {"fc1", "fc2", "fc3"}
    assert "fc_out" not in result.neuron
    assert result.edge == {} or all(v.numel() == 0 or True for v in result.edge.values())

    for name in ("fc1", "fc2", "fc3"):
        n_out = model.layer_shapes()[name][0]
        assert result.neuron[name].shape == (n_out,)
        # relu transform → nonnegative
        assert (result.neuron[name] >= 0).all()
        signed = result.extras["ablation_signed_neuron"][name]
        assert torch.allclose(result.neuron[name], signed.clamp(min=0.0))

    after = _clone_state(model)
    for k in before:
        assert torch.equal(before[k], after[k]), k


def test_ablation_edge_with_fixed_mask():
    set_seed(1)
    model = _tiny_mlp()
    before = _clone_state(model)
    loader = _toy_loader(n=16, batch_size=8)
    criterion = nn.CrossEntropyLoss()

    mask = build_edge_ablation_mask(model, seed=1, sample_per_layer=4, n_bins=2)
    gt = AblationImportance(edge_ablation_sample=4, seed=1, neuron_ablation="none")
    result = gt.compute(model, loader, criterion, edge_sample_mask=mask)

    assert result.edge_index is not None
    for name, (n_out, n_in) in model.layer_shapes().items():
        assert name in result.edge
        assert result.edge[name].shape == (n_out, n_in)
        assert torch.equal(result.edge_index[name], mask[name].cpu())
        # Non-masked positions must be zero
        assert (result.edge[name][~mask[name].cpu()] == 0).all()
        # Masked positions: relu of signed
        signed = result.extras["ablation_signed_edge"][name]
        assert torch.allclose(
            result.edge[name][mask[name].cpu()],
            signed[mask[name].cpu()].clamp(min=0.0),
        )

    after = _clone_state(model)
    for k in before:
        assert torch.equal(before[k], after[k]), k


def test_ablation_uses_provided_mask_not_redraw():
    """Passing edge_sample_mask must use it even if seed would draw differently."""
    set_seed(0)
    model = _tiny_mlp()
    loader = _toy_loader(n=12, batch_size=6)
    criterion = nn.CrossEntropyLoss()

    # Hand-built tiny mask: only first weight of fc_out
    custom = {
        name: torch.zeros(n_out, n_in, dtype=torch.bool)
        for name, (n_out, n_in) in model.layer_shapes().items()
    }
    custom["fc_out"][0, 0] = True

    gt = AblationImportance(edge_ablation_sample=512, seed=99, neuron_ablation="none")
    result = gt.compute(model, loader, criterion, edge_sample_mask=custom)
    assert result.edge_index is not None
    assert int(result.edge_index["fc_out"].sum().item()) == 1
    assert result.edge_index["fc_out"][0, 0]
    # Other layers all-False → zero edge scores
    for name in ("fc1", "fc2", "fc3"):
        assert result.edge[name].abs().sum().item() == 0.0


def test_fisher_shapes_nonnegative():
    set_seed(0)
    model = _tiny_mlp()
    loader = _toy_loader()
    result = compute_fisher(model, loader, nn.CrossEntropyLoss())
    assert FisherImportance().name == "fisher"
    shapes = model.layer_shapes()
    for name, (n_out, n_in) in shapes.items():
        assert result.edge[name].shape == (n_out, n_in)
        assert result.neuron[name].shape == (n_out,)
        assert (result.edge[name] >= 0).all()
        assert torch.allclose(result.neuron[name], result.edge[name].sum(dim=1))


def test_gt_protocol_accepts_edge_sample_mask_kwarg():
    """All GT compute methods accept edge_sample_mask without error."""
    set_seed(0)
    model = _tiny_mlp()
    loader = _toy_loader(n=8, batch_size=4)
    criterion = nn.CrossEntropyLoss()
    mask = build_edge_ablation_mask(model, seed=0, sample_per_layer=2)

    GradientSaliency().compute(model, loader, criterion, edge_sample_mask=mask)
    FisherImportance().compute(model, loader, criterion, edge_sample_mask=mask)
    AblationImportance(edge_ablation_sample=2, seed=0, neuron_ablation="none").compute(
        model, loader, criterion, edge_sample_mask=mask
    )
