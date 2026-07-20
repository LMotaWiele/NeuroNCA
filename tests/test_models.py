"""Tests for MLPReasoningNet: shapes, activation keys, layer_shapes, param count."""

from __future__ import annotations

import torch
import torch.nn as nn

from neuronca.models.mlp import MLPReasoningNet


def test_forward_logits_shape():
    net = MLPReasoningNet()
    x = torch.randn(8, 784)
    logits = net(x)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (8, 10)


def test_forward_accepts_image_shape():
    net = MLPReasoningNet()
    x = torch.randn(4, 1, 28, 28)
    logits = net(x)
    assert logits.shape == (4, 10)


def test_return_activations_keys_and_shapes():
    net = MLPReasoningNet()
    batch = 5
    x = torch.randn(batch, 784)
    logits, acts = net(x, return_activations=True)

    assert logits.shape == (batch, 10)
    expected_keys = {"act_in", "act_fc1", "act_fc2", "act_fc3", "act_fc_out"}
    assert set(acts.keys()) == expected_keys

    assert acts["act_in"].shape == (batch, 784)
    assert acts["act_fc1"].shape == (batch, 256)
    assert acts["act_fc2"].shape == (batch, 128)
    assert acts["act_fc3"].shape == (batch, 64)
    assert acts["act_fc_out"].shape == (batch, 10)

    # Logits key is the same tensor as returned logits (no ReLU on output).
    assert torch.equal(acts["act_fc_out"], logits)

    # Hidden activations are non-negative (ReLU).
    assert (acts["act_fc1"] >= 0).all()
    assert (acts["act_fc2"] >= 0).all()
    assert (acts["act_fc3"] >= 0).all()

    # last_activations mirrors the returned dict.
    assert net.last_activations is not None
    assert set(net.last_activations.keys()) == expected_keys


def test_named_linear_layers_order_and_types():
    net = MLPReasoningNet()
    layers = net.named_linear_layers()
    names = [n for n, _ in layers]
    assert names == ["fc1", "fc2", "fc3", "fc_out"]
    for name, mod in layers:
        assert isinstance(mod, nn.Linear)
        assert getattr(net, name) is mod


def test_layer_shapes_match_linear_modules():
    net = MLPReasoningNet()
    shapes = net.layer_shapes()
    assert set(shapes.keys()) == {"fc1", "fc2", "fc3", "fc_out"}
    for name, mod in net.named_linear_layers():
        n_out, n_in = shapes[name]
        assert n_out == mod.out_features
        assert n_in == mod.in_features
        assert mod.weight.shape == (n_out, n_in)

    assert shapes["fc1"] == (256, 784)
    assert shapes["fc2"] == (128, 256)
    assert shapes["fc3"] == (64, 128)
    assert shapes["fc_out"] == (10, 64)


def test_weight_parameter_count():
    """Weight-only count should be 784*256 + 256*128 + 128*64 + 64*10 = 242304."""
    net = MLPReasoningNet()
    weight_params = sum(m.weight.numel() for _, m in net.named_linear_layers())
    assert weight_params == 242_304

    # With biases: +256+128+64+10 = 242762 total Linear params.
    total_linear = sum(p.numel() for _, m in net.named_linear_layers() for p in m.parameters())
    assert total_linear == 242_762


def test_default_dims():
    net = MLPReasoningNet()
    assert net.input_dim == 784
    assert net.hidden == (256, 128, 64)
    assert net.num_classes == 10
    assert net.fc1.in_features == 784 and net.fc1.out_features == 256
    assert net.fc2.in_features == 256 and net.fc2.out_features == 128
    assert net.fc3.in_features == 128 and net.fc3.out_features == 64
    assert net.fc_out.in_features == 64 and net.fc_out.out_features == 10
