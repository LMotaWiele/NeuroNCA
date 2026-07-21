"""Unit tests for Phase 1a coordinate helpers and linear decay."""

from __future__ import annotations

import torch

from neuronca.plasticity.coords import (
    assign_hidden_coord,
    input_coords_28x28,
    linear_decay,
    output_coords,
    pairwise_distances,
)


def test_input_coords_shape_and_range():
    c = input_coords_28x28()
    assert c.shape == (784, 2)
    assert (c >= 0).all() and (c <= 1).all()
    # Corners: pixel 0 → (0,0), pixel 27 → (1,0) roughly.
    assert torch.allclose(c[0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(c[27], torch.tensor([1.0, 0.0]))


def test_output_coords():
    c = output_coords(10)
    assert c.shape == (10, 2)
    assert torch.allclose(c[0, 0], torch.tensor(0.0))
    assert torch.allclose(c[-1, 0], torch.tensor(1.0))


def test_linear_decay():
    d = torch.tensor([0.0, 0.5, 1.0, 1.5])
    out = linear_decay(d, radius=1.0)
    assert torch.allclose(out, torch.tensor([1.0, 0.5, 0.0, 0.0]))


def test_pairwise_and_assign():
    a = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    b = torch.tensor([[0.0, 0.0]])
    d = pairwise_distances(a, b)
    assert d.shape == (2, 1)
    assert torch.allclose(d[0, 0], torch.tensor(0.0))
    g = torch.Generator().manual_seed(0)
    c = assign_hidden_coord(generator=g)
    assert c.shape == (2,)
    assert (c >= 0).all() and (c <= 1).all()
