"""Optimizer state must not leak across slot prune → reactivate (Phase 1a §1)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import SGD, Adam

from neuronca.models.dynamic_mlp import DynamicMLPReasoningNet
from neuronca.plasticity.optimizer_reset import (
    reset_linear_edge_optimizer_state,
    reset_optimizer_state_for_indices,
)
from neuronca.training.optim import build_optimizer


def _tiny() -> DynamicMLPReasoningNet:
    return DynamicMLPReasoningNet(
        hidden_init=(4, 3, 2),
        hidden_max=(8, 6, 4),
        coord_seed=1,
    )


def test_reset_optimizer_state_for_indices_sgd_momentum():
    linear = nn.Linear(5, 4)
    opt = SGD(linear.parameters(), lr=0.1, momentum=0.9)
    x = torch.randn(2, 5)
    for _ in range(5):
        opt.zero_grad()
        linear(x).sum().backward()
        opt.step()
    st = opt.state[linear.weight]
    assert "momentum_buffer" in st
    assert st["momentum_buffer"].abs().sum() > 0
    # Scrub row 0.
    before_other = st["momentum_buffer"][1:].clone()
    reset_optimizer_state_for_indices(opt, linear.weight, row_indices=[0])
    assert st["momentum_buffer"][0].abs().sum() == 0
    assert torch.equal(st["momentum_buffer"][1:], before_other)


def test_reset_optimizer_state_adam_moments():
    linear = nn.Linear(5, 4)
    opt = Adam(linear.parameters(), lr=0.01)
    x = torch.randn(2, 5)
    for _ in range(5):
        opt.zero_grad()
        linear(x).sum().backward()
        opt.step()
    st = opt.state[linear.weight]
    assert st["exp_avg"].abs().sum() > 0
    reset_optimizer_state_for_indices(
        opt, linear.weight, row_indices=[1], col_indices=[2]
    )
    assert st["exp_avg"][1].abs().sum() == 0
    assert st["exp_avg"][:, 2].abs().sum() == 0
    assert st["exp_avg_sq"][1].abs().sum() == 0


def test_reactivate_neuron_scrubs_momentum():
    net = _tiny()
    opt = build_optimizer(net, "sgd", lr=0.1, momentum=0.9)
    x = torch.randn(8, 784)
    y = torch.randint(0, 10, (8,))

    for _ in range(8):
        opt.zero_grad()
        logits = net(x)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        net.zero_inactive_grads()
        opt.step()

    # Momentum on h1 slot 0 row of fc1 should be nonzero.
    st = opt.state[net.fc1.weight]
    assert "momentum_buffer" in st
    mom_before = st["momentum_buffer"][0].abs().sum().item()
    assert mom_before > 0

    net.begin_plasticity_step()
    net.deactivate_neuron("h1", 0)
    net.begin_plasticity_step()
    # Reactivate with optimizer scrub.
    net.activate_neuron("h1", 0, optimizer=opt)

    st2 = opt.state[net.fc1.weight]
    assert st2["momentum_buffer"][0].abs().sum().item() == 0
    # Bias row scrubbed.
    if net.fc1.bias is not None and net.fc1.bias in opt.state:
        assert opt.state[net.fc1.bias]["momentum_buffer"][0].abs().sum().item() == 0
    # Other active rows should still have momentum.
    assert st2["momentum_buffer"][1:4].abs().sum().item() > 0


def test_add_edge_scrubs_single_entry_momentum():
    net = _tiny()
    opt = build_optimizer(net, "sgd", lr=0.1, momentum=0.9)
    x = torch.randn(4, 784)
    y = torch.randint(0, 10, (4,))
    for _ in range(5):
        opt.zero_grad()
        nn.functional.cross_entropy(net(x), y).backward()
        net.zero_inactive_grads()
        opt.step()

    net.begin_plasticity_step()
    net.remove_edge("fc2", 0, 1)
    # Manually poison momentum at that index after prune (simulating leftover).
    buf = opt.state[net.fc2.weight]["momentum_buffer"]
    buf[0, 1] = 3.14
    net.begin_plasticity_step()
    net.add_edge("fc2", 0, 1, optimizer=opt)
    assert opt.state[net.fc2.weight]["momentum_buffer"][0, 1].item() == 0.0


def test_reset_edge_helper():
    w = nn.Parameter(torch.randn(3, 4))
    opt = SGD([w], lr=0.1, momentum=0.9)
    # Force state
    w.grad = torch.ones_like(w)
    opt.step()
    assert opt.state[w]["momentum_buffer"][1, 2].abs().item() > 0
    reset_linear_edge_optimizer_state(opt, w, 1, 2)
    assert opt.state[w]["momentum_buffer"][1, 2].item() == 0.0
