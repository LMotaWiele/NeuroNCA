"""Tests for DynamicMLPReasoningNet: masks, coords, grow/prune, cooldown."""

from __future__ import annotations

import torch
import torch.nn as nn

from neuronca.models.dynamic_mlp import DynamicMLPReasoningNet
from neuronca.plasticity.masks import (
    EDGE_ORIGIN_INCENTIVE,
    EDGE_ORIGIN_INIT,
    NEURON_ORIGIN_GROWN,
    NEURON_ORIGIN_INIT,
)

EXPECTED_ACT_KEYS = {"act_in", "act_fc1", "act_fc2", "act_fc3", "act_fc_out"}


def _small_net(**kwargs) -> DynamicMLPReasoningNet:
    defaults = dict(
        input_dim=784,
        hidden_init=(4, 3, 2),
        hidden_max=(8, 6, 4),
        num_classes=10,
        coord_seed=0,
    )
    defaults.update(kwargs)
    return DynamicMLPReasoningNet(**defaults)


def test_forward_shapes_and_activation_keys():
    net = _small_net()
    x = torch.randn(5, 784)
    logits = net(x)
    assert logits.shape == (5, 10)
    assert net.last_activations is not None
    assert set(net.last_activations.keys()) == EXPECTED_ACT_KEYS
    assert net.last_activations["act_fc1"].shape == (5, 8)  # capacity
    assert net.last_activations["act_fc2"].shape == (5, 6)
    assert net.last_activations["act_fc3"].shape == (5, 4)


def test_inactive_slots_zero_activation():
    net = _small_net()
    x = torch.randn(4, 784)
    logits, acts = net(x, return_activations=True)
    # Slots beyond init should be inactive → zero post-ReLU.
    assert (acts["act_fc1"][:, 4:] == 0).all()
    assert (acts["act_fc2"][:, 3:] == 0).all()
    assert (acts["act_fc3"][:, 2:] == 0).all()
    assert torch.isfinite(logits).all()


def test_init_dense_fully_connected_among_active():
    net = _small_net()
    # fc1: 4 active out × 784 in
    em = net.edge_mask("fc1")
    assert int(em[:4, :].sum()) == 4 * 784
    assert int(em[4:, :].sum()) == 0
    assert int((net.edge_origin("fc1")[:4, :] == EDGE_ORIGIN_INIT).sum()) == 4 * 784
    assert net.hidden_counts() == {"h1": 4, "h2": 3, "h3": 2}
    counts = net.edge_counts(by_origin=True)
    assert counts["fc1"]["init_dense"] == 4 * 784
    assert counts["fc1"]["incentive_grown"] == 0


def test_coords_assigned_at_birth_nan_when_inactive():
    net = _small_net()
    c1 = net.coords("h1")
    assert torch.isfinite(c1[:4]).all()
    assert torch.isnan(c1[4:]).all()
    # Input on unit square
    assert net.coords_in.shape == (784, 2)
    assert (net.coords_in >= 0).all() and (net.coords_in <= 1).all()


def test_deactivate_and_reactivate_fresh_coord():
    net = _small_net()
    net.begin_plasticity_step()
    old = net.coords("h1")[0].clone()
    info = net.deactivate_neuron("h1", 0)
    assert info["neuron_origin"] == NEURON_ORIGIN_INIT
    assert not bool(net.active_h1[0])
    assert torch.isnan(net.coords("h1")[0]).all()
    # Same-step reactivate must fail (cooldown).
    try:
        net.activate_neuron("h1", 0)
        raise AssertionError("expected cooldown RuntimeError")
    except RuntimeError as e:
        assert "cooldown" in str(e).lower()
    # Next plasticity step: fresh coord allowed.
    net.begin_plasticity_step()
    info2 = net.activate_neuron("h1", 0, origin=NEURON_ORIGIN_GROWN)
    assert info2["neuron_origin"] == NEURON_ORIGIN_GROWN
    assert bool(net.active_h1[0])
    new = net.coords("h1")[0]
    assert torch.isfinite(new).all()
    # Extremely unlikely to redraw identical coord; allow equality but require finite.
    assert not torch.isnan(new).any()


def test_remove_and_add_edge_provenance_and_cooldown():
    net = _small_net()
    net.begin_plasticity_step()
    ev = net.remove_edge("fc2", 0, 0)
    assert ev["origin"] == EDGE_ORIGIN_INIT
    assert not bool(net.edge_mask("fc2")[0, 0])
    try:
        net.add_edge("fc2", 0, 0)
        raise AssertionError("expected cooldown on same-step edge reform")
    except RuntimeError:
        pass
    net.begin_plasticity_step()
    ev2 = net.add_edge("fc2", 0, 0, origin=EDGE_ORIGIN_INCENTIVE)
    assert ev2["origin"] == EDGE_ORIGIN_INCENTIVE
    assert bool(net.edge_mask("fc2")[0, 0])
    assert int(net.edge_origin("fc2")[0, 0].item()) == EDGE_ORIGIN_INCENTIVE


def test_masked_forward_matches_zeroed_inactive_weights():
    """Inactive path contributes nothing: zeroing unused weights does not change logits."""
    net = _small_net()
    x = torch.randn(3, 784)
    y1 = net(x).detach().clone()
    with torch.no_grad():
        # Corrupt inactive capacity weights heavily.
        net.fc1.weight[4:] = 100.0
        net.fc2.weight[3:] = 100.0
        net.fc3.weight[2:] = 100.0
    y2 = net(x)
    assert torch.allclose(y1, y2, atol=1e-5)


def test_zero_inactive_grads():
    torch.manual_seed(0)
    net = _small_net()
    # Dense CE-style loss on logits avoids dead-ReLU all-zero grad flukes.
    x = torch.randn(16, 784)
    logits = net(x)
    loss = logits.pow(2).mean()
    loss.backward()
    assert net.fc1.weight.grad is not None
    grad_active_before = net.fc1.weight.grad[:4].abs().sum().item()
    assert grad_active_before > 0
    # Poison inactive grads then scrub.
    net.fc1.weight.grad[4:] = 1.0
    net.fc1.bias.grad[4:] = 1.0
    net.zero_inactive_grads()
    assert (net.fc1.weight.grad[4:] == 0).all()
    assert (net.fc1.bias.grad[4:] == 0).all()
    assert net.fc1.weight.grad[:4].abs().sum().item() > 0


def test_active_parameter_count_less_than_capacity():
    net = _small_net()
    assert net.active_parameter_count() < net.capacity_parameter_count()
    # Dense init among actives: roughly init-sized dense MLP.
    assert net.active_parameter_count() > 0


def test_image_input_shape():
    net = _small_net()
    x = torch.randn(2, 1, 28, 28)
    logits = net(x)
    assert logits.shape == (2, 10)
