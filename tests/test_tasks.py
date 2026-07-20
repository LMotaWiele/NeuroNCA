"""Unit tests for task transforms, TaskContext, and MNIST probe split."""

from __future__ import annotations

import torch

from neuronca.data.mnist import (
    build_mnist_datasets,
    build_mnist_loaders,
    split_train_probe_indices,
)
from neuronca.data.tasks import (
    IdentityTransform,
    TaskContext,
    make_identity_transform,
    make_label_permutation_transform,
    make_pixel_permutation_transform,
    make_task_transform,
)
from neuronca.io.seed import set_seed


def test_set_seed_is_deterministic():
    set_seed(42)
    a = torch.rand(4)
    set_seed(42)
    b = torch.rand(4)
    assert torch.equal(a, b)


def test_label_permutation_factory_is_seeded():
    t1 = make_label_permutation_transform(num_classes=10, seed=123)
    t2 = make_label_permutation_transform(num_classes=10, seed=123)
    t3 = make_label_permutation_transform(num_classes=10, seed=999)
    assert torch.equal(t1.permutation, t2.permutation)
    assert not torch.equal(t1.permutation, t3.permutation)
    # π must be a permutation of 0..9
    assert torch.equal(torch.sort(t1.permutation).values, torch.arange(10))


def test_probe_labels_under_b1_equal_pi_raw():
    """Probe labels under B1 equal π(raw labels) for shared indices."""
    ctx = TaskContext(IdentityTransform())
    bundles = build_mnist_datasets(
        ctx,
        root="data/mnist",
        probe_size=128,  # smaller for test speed; same logic as 1024
        seed=0,
        download=True,
    )
    probe_indices = bundles["probe_indices"]
    train_full = bundles["train_full"]
    raw_targets = train_full.targets
    if not torch.is_tensor(raw_targets):
        raw_targets = torch.tensor(raw_targets)
    raw_targets = raw_targets.long()

    b1 = make_label_permutation_transform(num_classes=10, seed=123)
    pi = b1.permutation
    ctx.set_transform(b1)

    probe_ds = bundles["probe"]
    # Walk probe dataset in order; Subset preserves probe_indices order
    for i in range(len(probe_ds)):
        _x, y_mapped = probe_ds[i]
        raw_y = raw_targets[probe_indices[i]]
        expected = pi[raw_y]
        assert int(y_mapped) == int(expected), (
            f"index {i}: mapped={int(y_mapped)} expected=π({int(raw_y)})={int(expected)}"
        )


def test_task_context_atomic_flip_affects_train_and_probe():
    """TaskContext atomic flip affects both train and probe views."""
    ctx = TaskContext(make_identity_transform())
    bundles = build_mnist_datasets(
        ctx,
        root="data/mnist",
        probe_size=64,
        seed=1,
        download=True,
    )
    train_ds = bundles["train"]
    probe_ds = bundles["probe"]
    train_full = bundles["train_full"]
    train_indices = bundles["train_indices"]
    probe_indices = bundles["probe_indices"]
    raw_targets = train_full.targets
    if not torch.is_tensor(raw_targets):
        raw_targets = torch.tensor(raw_targets)
    raw_targets = raw_targets.long()

    # Under identity, labels match raw
    _, y_train_a = train_ds[0]
    _, y_probe_a = probe_ds[0]
    assert int(y_train_a) == int(raw_targets[train_indices[0]])
    assert int(y_probe_a) == int(raw_targets[probe_indices[0]])

    b1 = make_label_permutation_transform(num_classes=10, seed=123)
    pi = b1.permutation
    # Atomic flip — single call
    ctx.set_transform(b1)

    _, y_train_b = train_ds[0]
    _, y_probe_b = probe_ds[0]
    assert int(y_train_b) == int(pi[raw_targets[train_indices[0]]])
    assert int(y_probe_b) == int(pi[raw_targets[probe_indices[0]]])
    # Both views read the same transform instance
    assert train_ds.task_context is probe_ds.task_context is ctx
    assert ctx.transform is b1


def test_pixel_permutation_is_deterministic_given_seed():
    """Pixel perm is deterministic given seed."""
    t1 = make_pixel_permutation_transform(num_pixels=28 * 28, seed=7)
    t2 = make_pixel_permutation_transform(num_pixels=28 * 28, seed=7)
    t3 = make_pixel_permutation_transform(num_pixels=28 * 28, seed=8)
    assert torch.equal(t1.permutation, t2.permutation)
    assert not torch.equal(t1.permutation, t3.permutation)

    x = torch.arange(28 * 28, dtype=torch.float32).reshape(1, 28, 28)
    y1 = t1.map_input(x)
    y2 = t2.map_input(x.clone())
    assert torch.equal(y1, y2)
    # Round-trip via inverse perm recovers original
    inv = torch.empty_like(t1.permutation)
    inv[t1.permutation] = torch.arange(t1.permutation.numel())
    recovered = y1.reshape(-1)[inv].reshape(1, 28, 28)
    assert torch.equal(recovered, x)


def test_make_task_transform_dispatch():
    a = make_task_transform("identity")
    assert isinstance(a, IdentityTransform)
    b1 = make_task_transform("label_permutation", seed=123)
    assert torch.equal(
        b1.permutation,
        make_label_permutation_transform(seed=123).permutation,
    )
    b3 = make_task_transform("pixel_permutation", seed=5)
    assert torch.equal(
        b3.permutation,
        make_pixel_permutation_transform(seed=5).permutation,
    )


def test_probe_train_split_disjoint_and_sized():
    n = 1000
    targets = torch.arange(n) % 10
    train_idx, probe_idx = split_train_probe_indices(
        n, probe_size=100, seed=0, targets=targets, num_classes=10
    )
    assert len(probe_idx) == 100
    assert len(train_idx) == 900
    probe_set = set(probe_idx.tolist())
    train_set = set(train_idx.tolist())
    assert probe_set.isdisjoint(train_set)
    assert probe_set | train_set == set(range(n))


def test_build_mnist_loaders_smoke():
    ctx = TaskContext(make_identity_transform())
    out = build_mnist_loaders(
        ctx,
        root="data/mnist",
        probe_size=32,
        seed=0,
        batch_size=16,
        num_workers=0,
        download=True,
    )
    x, y = next(iter(out["probe_loader"]))
    assert x.shape[0] == 16
    assert x.shape[1] == 1  # MNIST channel
    assert y.shape[0] == 16
    assert out["task_context"] is ctx
