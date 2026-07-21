"""Tests for snapshot-scoped probe cache: version, integrity, GT equivalence."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neuronca.data.mnist import build_mnist_loaders
from neuronca.data.tasks import (
    TaskContext,
    make_identity_transform,
    make_label_permutation_transform,
)
from neuronca.importance.ablation import AblationImportance
from neuronca.importance.base import mean_probe_loss
from neuronca.importance.gradient_saliency import GradientSaliency
from neuronca.importance.probe_cache import (
    ProbeCache,
    ProbeCacheIntegrityError,
    ProbeCacheVersionError,
    assert_probe_cache_matches_loader,
    extract_task_context,
    materialize_probe_cache,
)
from neuronca.io.seed import set_seed
from neuronca.models.mlp import MLPReasoningNet
from neuronca.training.callbacks import InstrumentedTrainer
from neuronca.training.optim import build_optimizer


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


def test_materialize_bit_equal_to_two_loader_passes():
    set_seed(0)
    ctx = TaskContext(make_identity_transform())
    loaders = build_mnist_loaders(
        ctx, probe_size=128, seed=0, batch_size=32, num_workers=0
    )
    probe = loaders["probe_loader"]
    device = torch.device("cpu")
    cache = materialize_probe_cache(probe, device=device)
    assert extract_task_context(probe) is ctx
    assert cache.task_version == ctx.version
    assert cache.task_context is ctx

    xs, ys = [], []
    for x, y in probe:
        xs.append(x)
        ys.append(y)
    X = torch.cat(xs, dim=0)
    Y = torch.cat(ys, dim=0)
    assert torch.equal(cache.X.cpu(), X)
    assert torch.equal(cache.Y.cpu(), Y)


def test_version_guard_raises_on_set_transform():
    ctx = TaskContext(make_identity_transform())
    loaders = build_mnist_loaders(
        ctx, probe_size=64, seed=1, batch_size=32, num_workers=0
    )
    cache = materialize_probe_cache(loaders["probe_loader"], device="cpu")
    cache.ensure_version()  # ok
    ctx.set_transform(make_label_permutation_transform(seed=99))
    assert cache.task_version != ctx.version
    try:
        cache.ensure_version()
        raised = False
    except ProbeCacheVersionError:
        raised = True
    assert raised


def test_integrity_detects_mutated_x():
    loader = _toy_loader()
    cache = materialize_probe_cache(loader, device="cpu")
    assert_probe_cache_matches_loader(cache, loader, device="cpu")
    # Corrupt shared storage
    cache.X[0, 0] = 999.0
    try:
        assert_probe_cache_matches_loader(cache, loader, device="cpu")
        raised = False
    except ProbeCacheIntegrityError:
        raised = True
    assert raised


def test_mean_probe_loss_cache_matches_loader():
    set_seed(0)
    model = _tiny_mlp()
    loader = _toy_loader()
    crit = nn.CrossEntropyLoss()
    cache = materialize_probe_cache(loader, device="cpu")
    l_loader = mean_probe_loss(model, loader, crit, device=torch.device("cpu"))
    l_cache = mean_probe_loss(model, cache, crit, device=torch.device("cpu"))
    assert abs(l_loader - l_cache) < 1e-6


def test_grad_and_ablation_match_loader_vs_cache():
    """Identical GT tensors: live DataLoader vs ProbeCache on same model."""
    set_seed(42)
    model = _tiny_mlp()
    loader = _toy_loader(n=24, batch_size=6)
    crit = nn.CrossEntropyLoss()
    cache = materialize_probe_cache(loader, device="cpu")

    g_loader = GradientSaliency().compute(model, loader, crit)
    g_cache = GradientSaliency().compute(model, cache, crit)
    for name in g_loader.edge:
        assert torch.allclose(g_loader.edge[name], g_cache.edge[name], atol=1e-6), name
        assert torch.allclose(
            g_loader.neuron[name], g_cache.neuron[name], atol=1e-6
        ), name

    mask = {
        name: torch.zeros(n_out, n_in, dtype=torch.bool)
        for name, (n_out, n_in) in model.layer_shapes().items()
    }
    # A few edges + all neurons
    mask["fc1"][0, 0] = True
    mask["fc2"][0, 0] = True
    abl = AblationImportance(
        edge_ablation_sample=2,
        neuron_ablation="all_hidden",
        seed=0,
        hidden_layers=("fc1", "fc2", "fc3"),
    )
    a_loader = abl.compute(model, loader, crit, edge_sample_mask=mask)
    a_cache = abl.compute(model, cache, crit, edge_sample_mask=mask)
    for name in a_loader.neuron:
        assert torch.allclose(
            a_loader.neuron[name], a_cache.neuron[name], atol=1e-6
        ), name
    for name in a_loader.edge:
        assert torch.allclose(
            a_loader.edge[name], a_cache.edge[name], atol=1e-6
        ), name


def test_run_gt_builds_fresh_cache_and_integrity_passes():
    set_seed(0)
    model = _tiny_mlp()
    opt = build_optimizer(model)
    loader = _toy_loader()
    trainer = InstrumentedTrainer(model, opt, signals=[], device="cpu")
    results = trainer.run_gt(
        [GradientSaliency()],
        loader,
        verify_probe_cache=True,
    )
    assert "grad_abs" in results
    assert results["grad_abs"].edge


def test_snapshot_cache_not_shared_across_task_flip():
    """Each run_gt materializes under the current TaskContext version."""
    set_seed(0)
    ctx = TaskContext(make_identity_transform())
    loaders = build_mnist_loaders(
        ctx, probe_size=64, seed=0, batch_size=32, num_workers=0
    )
    probe = loaders["probe_loader"]
    model = MLPReasoningNet()
    opt = build_optimizer(model)
    trainer = InstrumentedTrainer(model, opt, signals=[], device="cpu")

    r1 = trainer.run_gt([GradientSaliency()], probe, verify_probe_cache=True)
    v1 = ctx.version
    ctx.set_transform(make_label_permutation_transform(seed=7))
    assert ctx.version != v1
    r2 = trainer.run_gt([GradientSaliency()], probe, verify_probe_cache=True)
    # Different task → labels/loss surface differ → GT maps should change
    # (not a hard requirement on every layer, but overall should not be identical).
    same = all(
        torch.equal(r1["grad_abs"].edge[k], r2["grad_abs"].edge[k])
        for k in r1["grad_abs"].edge
    )
    assert not same, "GT after label flip should not match identity-task GT"
