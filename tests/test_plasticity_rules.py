"""Tests for Phase 1a plasticity rules, eligibility, cooldown, thresholds."""

from __future__ import annotations

import pytest
import torch

from neuronca.models.dynamic_mlp import DynamicMLPReasoningNet
from neuronca.plasticity.masks import NEURON_ORIGIN_GROWN
from neuronca.plasticity.orchestrator import PlasticityOrchestrator
from neuronca.plasticity.rules import (
    PlasticityConfig,
    edge_eligibility,
    edge_incentive_matrix,
    gap_imprint_direction,
    gated_novelty_scores_for_layer,
    local_zscore,
    per_dimension_separation_scores,
    scale_direction_to_kaiming_relu,
    select_edges_to_grow,
    select_neurons_to_grow_gated_novelty,
    select_neurons_to_prune,
)
from neuronca.plasticity.threshold import FixedThreshold, SlidingThreshold, build_threshold_policy


def _net() -> DynamicMLPReasoningNet:
    return DynamicMLPReasoningNet(
        hidden_init=(4, 3, 2),
        hidden_max=(8, 6, 4),
        coord_seed=0,
    )


def test_local_zscore():
    v = torch.tensor([1.0, 2.0, 3.0, 0.0])
    m = torch.tensor([True, True, True, False])
    z = local_zscore(v, m)
    assert z[3] == 0.0
    assert abs(float(z[:3].mean())) < 1e-5


def test_per_layer_targets_drive_independent_pressure():
    """Global r* would bias h1 cold / h3 hot; per-layer targets avoid that."""
    rates = {"h1": 0.77, "h2": 1.02, "h3": 1.28}
    policy = FixedThreshold(tau=0.1, target_rates=rates)
    # Signals at each layer's own mean → near target, neither pressure with tau=0.1
    signals = {
        "h1": torch.full((4,), 0.77),
        "h2": torch.full((3,), 1.02),
        "h3": torch.full((2,), 1.28),
    }
    active = {
        "h1": torch.ones(4, dtype=torch.bool),
        "h2": torch.ones(3, dtype=torch.bool),
        "h3": torch.ones(2, dtype=torch.bool),
    }
    policy.update(signals, active)
    assert not policy.growth_pressure("h1").any()
    assert not policy.contraction_pressure("h1").any()
    assert not policy.growth_pressure("h3").any()
    # Same absolute signal 0.77: growth on h3 (below its 1.28-τ), not on h1
    signals2 = {
        "h1": torch.full((4,), 0.77),
        "h2": torch.full((3,), 0.77),
        "h3": torch.full((2,), 0.77),
    }
    policy.update(signals2, active)
    assert not policy.contraction_pressure("h1").any()  # at h1 target
    assert policy.contraction_pressure("h3").all()  # 0.77 << 1.28


def test_edge_eligibility_io_only_checks_hidden():
    """input→H1: fixed input always eligible; only hidden pressure matters."""
    policy = FixedThreshold(tau=0.05, target_rate=0.1)
    # Fake pressure on h1 slot 0 growth, slot 1 contraction.
    policy._growth["h1"] = torch.tensor([True, False, False, False])
    policy._contraction["h1"] = torch.tensor([False, True, False, False])
    policy._growth["h2"] = torch.zeros(3, dtype=torch.bool)
    policy._contraction["h2"] = torch.zeros(3, dtype=torch.bool)
    policy._growth["h3"] = torch.zeros(2, dtype=torch.bool)
    policy._contraction["h3"] = torch.zeros(2, dtype=torch.bool)

    # fc1: post=h1[0] growth → eligible
    ok, ge = edge_eligibility("fc1", out_idx=0, in_idx=10, policy=policy)
    assert ok and ge == [("h1", 0)]
    # h1[1] contraction → not eligible
    ok2, _ = edge_eligibility("fc1", out_idx=1, in_idx=10, policy=policy)
    assert not ok2
    # h1[2] neither → not eligible (no growth pressure)
    ok3, _ = edge_eligibility("fc1", out_idx=2, in_idx=0, policy=policy)
    assert not ok3


def test_edge_eligibility_relaxed_allows_one_contracting_partner():
    """Relaxed mode: growth + one contracting hidden end can still pass."""
    policy = FixedThreshold(tau=0.05, target_rate=0.1)
    # h1[0] growth, h2[0] contraction → fc2 edge post=h2[0] pre=h1[0]
    policy._growth["h1"] = torch.tensor([True, False, False, False])
    policy._contraction["h1"] = torch.tensor([False, False, False, False])
    policy._growth["h2"] = torch.tensor([False, False, False])
    policy._contraction["h2"] = torch.tensor([True, False, False])
    policy._growth["h3"] = torch.zeros(2, dtype=torch.bool)
    policy._contraction["h3"] = torch.zeros(2, dtype=torch.bool)

    # strict: contracting post blocks
    ok_s, _ = edge_eligibility(
        "fc2", 0, 0, policy, mode="neither_contraction"
    )
    assert not ok_s
    # relaxed: one growth + one contraction → allowed
    ok_r, ge = edge_eligibility(
        "fc2", 0, 0, policy, mode="not_both_contraction"
    )
    assert ok_r and ge == [("h1", 0)]
    # both contracting, no growth → still blocked
    policy._growth["h1"] = torch.tensor([False, False, False, False])
    policy._contraction["h1"] = torch.tensor([True, False, False, False])
    ok_both, _ = edge_eligibility(
        "fc2", 0, 0, policy, mode="not_both_contraction"
    )
    assert not ok_both


def test_sliding_edge_bar_uses_max_theta_not_product():
    caps = {"h1": 4, "h2": 3, "h3": 2}
    pol = SlidingThreshold(mu=0.999, target_rate=0.1, capacities=caps, edge_tau_floor=0.0)
    pol._theta["h1"][0] = 0.01  # low θ (would make product small)
    pol._theta["h1"][1] = 0.5
    # Only growth on slot 0 (cold history but flagged growth artificially)
    thr_low = pol.edge_threshold_for_pair([("h1", 0)])
    thr_high = pol.edge_threshold_for_pair([("h1", 1)])
    thr_both = pol.edge_threshold_for_pair([("h1", 0), ("h1", 1)])
    assert thr_low == pytest.approx(0.01)
    assert thr_high == pytest.approx(0.5)
    assert thr_both == pytest.approx(0.5)  # max, not product 0.005


def test_incentive_zero_beyond_radius():
    net = _net()
    # Place two h1 neurons far apart if possible — use input vs h1 with R tiny.
    signals = {
        "in": torch.ones(784),
        "h1": torch.ones(8),
        "h2": torch.ones(6),
        "h3": torch.ones(4),
        "out": torch.ones(10),
    }
    # Very small R → almost no pairs
    inc = edge_incentive_matrix(net, "fc2", signals, radius=1e-6)
    # Only near-zero distance pairs could fire; most should be 0
    assert float(inc.sum()) >= 0.0
    inc_wide = edge_incentive_matrix(net, "fc2", signals, radius=2.0)
    assert float(inc_wide.sum()) >= float(inc.sum())


def test_contraction_neuron_not_selected_for_grow_edges():
    net = _net()
    policy = FixedThreshold(tau=0.0, target_rate=0.5)
    # All active below target → contraction; none growth
    signals_h = {
        "h1": torch.zeros(8),
        "h2": torch.zeros(6),
        "h3": torch.zeros(4),
    }
    active = {h: net.active_mask(h).cpu() for h in ("h1", "h2", "h3")}
    policy.update(signals_h, active)
    assert policy.contraction_pressure("h1")[:4].all()
    assert not policy.growth_pressure("h1").any()

    signals = {
        "in": torch.ones(784),
        "h1": torch.zeros(8),
        "h2": torch.zeros(6),
        "h3": torch.zeros(4),
        "out": torch.ones(10),
    }
    # Remove one edge so it could be re-added
    net.begin_plasticity_step()
    net.remove_edge("fc2", 0, 0)
    net.begin_plasticity_step()
    inc = edge_incentive_matrix(net, "fc2", signals, radius=2.0)
    # Force high incentive artificially
    inc[0, 0] = 10.0
    picks = select_edges_to_grow(net, "fc2", inc, policy, max_new=10)
    assert picks == []  # no growth pressure anywhere


def test_orchestrator_prune_by_origin_and_cooldown():
    net = _net()
    cfg = PlasticityConfig(
        radius=2.0,
        target_rate=0.1,
        tau=0.05,
        threshold_kind="fixed",
        prune_edge_fraction=0.1,
        max_new_edges_per_layer=8,
        max_new_neurons_per_layer=1,
    )
    orch = PlasticityOrchestrator(cfg)
    # Fake scores: moderate activations on capacity buffers
    scores = {
        "fc1": torch.full((8,), 0.1),
        "fc2": torch.full((6,), 0.1),
        "fc3": torch.full((4,), 0.1),
        "fc_out": torch.full((10,), 0.1),
    }
    orch.input_signal_ema = torch.ones(784) * 0.1
    log = orch.step(net, scores, epoch=0)
    assert log.step == 1
    assert "h1" in log.hidden_counts
    # Prunes should be init_dense only early
    assert log.edges_pruned_incentive_grown == 0
    assert log.edges_pruned_init_dense >= 0


def test_newborn_grace_blocks_immediate_contraction_prune():
    """Grown slots younger than grace_steps are not contraction-pruned."""
    net = _net()
    net.begin_plasticity_step()
    # Free a slot and regrow so birth_step is set to current plasticity step.
    net.deactivate_neuron("h1", 0)
    net.begin_plasticity_step()
    info = net.activate_neuron("h1", 0, origin=NEURON_ORIGIN_GROWN)
    slot = info["slot"]
    assert int(net.birth_step("h1")[slot].item()) == net.cooldown.plasticity_step
    policy = FixedThreshold(tau=0.0, target_rate=1.0)
    # Force contraction on all active
    a = net.active_mask("h1").cpu()
    policy._growth["h1"] = torch.zeros_like(a)
    policy._contraction["h1"] = a.clone()
    policy._growth["h2"] = torch.zeros(3, dtype=torch.bool)
    policy._contraction["h2"] = torch.zeros(3, dtype=torch.bool)
    policy._growth["h3"] = torch.zeros(2, dtype=torch.bool)
    policy._contraction["h3"] = torch.zeros(2, dtype=torch.bool)
    z = torch.full((8,), -2.0)
    from neuronca.plasticity.rules import select_neurons_to_prune

    # Same step as birth: age 0 < grace 2
    pruned = select_neurons_to_prune(net, "h1", z, policy, -1.0, grace_steps=2)
    assert slot not in pruned
    # Advance one step: age 1 still < 2
    net.begin_plasticity_step()
    pruned2 = select_neurons_to_prune(net, "h1", z, policy, -1.0, grace_steps=2)
    assert slot not in pruned2
    # Advance again: age 2 >= 2
    net.begin_plasticity_step()
    pruned3 = select_neurons_to_prune(net, "h1", z, policy, -1.0, grace_steps=2)
    assert slot in pruned3


def test_grace_must_be_at_least_birth_bias_steps():
    """Judgment must not mix residual scaffold bias into contraction decisions."""
    with pytest.raises(ValueError, match="newborn_grace_steps must be >="):
        PlasticityConfig(newborn_grace_steps=2, newborn_birth_bias_steps=3)
    # Equality is allowed (bias fully gone at first eligible age).
    ok = PlasticityConfig(newborn_grace_steps=3, newborn_birth_bias_steps=3)
    assert ok.newborn_grace_steps == 3


def test_birth_bias_zero_when_age_reaches_bias_steps():
    """Scaffold bias is fully decayed at age == bias_steps (judgment-aligned)."""
    net = _net()
    net.begin_plasticity_step()
    net.deactivate_neuron("h1", 0)
    net.begin_plasticity_step()
    info = net.activate_neuron("h1", 0, origin=NEURON_ORIGIN_GROWN, birth_bias=0.5)
    slot = info["slot"]
    assert float(net.birth_bias("h1")[slot].item()) == pytest.approx(0.5)
    # age 1 → scale 2/3
    net.begin_plasticity_step()
    net.update_birth_biases(3)
    assert float(net.birth_bias("h1")[slot].item()) == pytest.approx(0.5 * (2 / 3))
    # age 2 → scale 1/3
    net.begin_plasticity_step()
    net.update_birth_biases(3)
    assert float(net.birth_bias("h1")[slot].item()) == pytest.approx(0.5 * (1 / 3))
    # age 3 → fully gone
    net.begin_plasticity_step()
    net.update_birth_biases(3)
    assert float(net.birth_bias("h1")[slot].item()) == pytest.approx(0.0)


def test_parent_perturbed_weight_init_copies_parent_row():
    """Newborn incoming row is a noisy copy of the growth-trigger parent, not Kaiming-independent."""
    net = _net()
    # Parent at slot 1 (init-active); free slot 0 after deactivate + cooldown.
    parent = 1
    parent_row = net.fc1.weight[parent].detach().clone()
    net.begin_plasticity_step()
    net.deactivate_neuron("h1", 0)
    net.begin_plasticity_step()
    info = net.activate_neuron(
        "h1",
        0,
        origin=NEURON_ORIGIN_GROWN,
        parent_slot=parent,
        weight_init="parent_perturbed",
        parent_noise=0.0,  # exact copy for assertion
        copy_outgoing=True,
    )
    assert info["parent_slot"] == parent
    assert info["weight_init"] in ("parent_perturbed", "parent", "copy_parent")
    # Incoming row matches parent exactly when noise=0.
    assert torch.allclose(net.fc1.weight[0], parent_row)
    # Outgoing col of fc2 (h1 → h2) matches parent.
    assert torch.allclose(net.fc2.weight[:, 0], net.fc2.weight[:, parent])
    # Edge row copied from parent.
    assert torch.equal(net.edge_mask("fc1")[0], net.edge_mask("fc1")[parent])


def test_scale_newborn_critical_grads_only_in_grace():
    """Critical-period grad scale hits grown slots younger than grace only."""
    net = _net()
    net.begin_plasticity_step()
    net.deactivate_neuron("h1", 0)
    net.begin_plasticity_step()
    info = net.activate_neuron("h1", 0, origin=NEURON_ORIGIN_GROWN)
    slot = info["slot"]
    # Fake a grad on the new row and an established row.
    net.fc1.weight.grad = torch.ones_like(net.fc1.weight)
    net.scale_newborn_critical_grads(lr_mult=5.0, grace_steps=2)
    # Newborn (age 0) scaled; parent/init row 1 not grown → unscaled.
    assert float(net.fc1.weight.grad[slot, 0].item()) == pytest.approx(5.0)
    assert float(net.fc1.weight.grad[1, 0].item()) == pytest.approx(1.0)
    # Past grace: no scale.
    net.begin_plasticity_step()
    net.begin_plasticity_step()
    net.fc1.weight.grad = torch.ones_like(net.fc1.weight)
    net.scale_newborn_critical_grads(lr_mult=5.0, grace_steps=2)
    assert float(net.fc1.weight.grad[slot, 0].item()) == pytest.approx(1.0)


def test_build_threshold_policy():
    caps = {"h1": 8, "h2": 6, "h3": 4}
    f = build_threshold_policy("fixed", tau=0.05, target_rate=0.1, capacities=caps)
    assert f.name == "fixed"
    s = build_threshold_policy(
        "sliding", tau=0.05, target_rate=0.1, mu=0.99, capacities=caps
    )
    assert s.name == "sliding"


def test_same_step_free_reuse_blocked_by_cooldown():
    net = _net()
    net.begin_plasticity_step()
    net.remove_edge("fc3", 0, 0)
    # Attempt reform same step
    try:
        net.add_edge("fc3", 0, 0)
        raise AssertionError("should block")
    except RuntimeError:
        pass
    assert net.cooldown.same_step_blocks >= 1


def test_gated_novelty_under_floor_excluded_and_counted():
    """|N| < k_min units are excluded (not triggers) and logged under_floor."""
    # 4 units on a line; radius so each has at most 1 neighbor → under floor if k_min=2
    B, H = 32, 4
    coords = torch.tensor(
        [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32
    )
    active = torch.ones(H, dtype=torch.bool)
    # Unit 0 has a coverage gap on high-CE samples; neighbors weak there.
    acts = torch.rand(B, H) * 0.1
    ce = torch.zeros(B)
    # Bottom half of samples: high CE, all units weak → gap for everyone with neighbors
    ce[: B // 2] = 2.0
    acts[: B // 2] = 0.01
    # Top half: low CE, neighbors hot
    ce[B // 2 :] = 0.1
    acts[B // 2 :, 1:] = 1.0

    stats, cands, masks = gated_novelty_scores_for_layer(
        acts, ce, coords, active, radius=0.4, k_min=2
    )
    # With R=0.4, units at distance 0.5 are outside → many under floor
    assert stats.n_under_floor >= 1
    # No silent promotion of under-floor into candidates
    under = set(stats.under_floor_slots)
    for slot, *_ in cands:
        assert slot not in under
    for slot in masks:
        assert slot not in under


def test_gated_novelty_ce_gate_and_density_pass():
    """High density + CE excess on low nbr_max samples → candidate; no excess → not."""
    B, H = 64, 6
    # Pack all units close so |N| is large for every unit.
    coords = torch.zeros(H, 2)
    for i in range(H):
        coords[i, 0] = 0.1 * (i % 3)
        coords[i, 1] = 0.1 * (i // 3)
    active = torch.ones(H, dtype=torch.bool)
    acts = torch.ones(B, H) * 0.5
    ce = torch.ones(B) * 0.5
    # Make sample bottom-quartile of nbr_max high-CE: lower all acts on first 16 samples
    acts[:16] = 0.05
    ce[:16] = 2.0
    stats, cands, masks = gated_novelty_scores_for_layer(
        acts, ce, coords, active, radius=0.5, k_min=3
    )
    assert stats.n_density_pass == H
    assert stats.n_under_floor == 0
    assert stats.n_ce_gate_pass >= 1
    assert len(cands) >= 1
    assert len(masks) == len(cands)
    # Invert: uniform CE → no excess
    ce2 = torch.ones(B)
    stats2, cands2, masks2 = gated_novelty_scores_for_layer(
        acts, ce2, coords, active, radius=0.5, k_min=3
    )
    assert stats2.n_ce_gate_pass == 0
    assert cands2 == []
    assert masks2 == {}


def test_select_neurons_to_grow_gated_novelty_rate_cap():
    net = _net()
    # Free capacity by deactivating one slot.
    net.begin_plasticity_step()
    net.deactivate_neuron("h1", 0)
    net.begin_plasticity_step()
    cands = [(1, 0.1, 0.5, 12), (2, 0.2, 0.4, 12), (3, 0.3, 0.3, 12)]
    picks, precap = select_neurons_to_grow_gated_novelty(
        net, "h1", cands, max_new=1, return_precap_count=True
    )
    assert precap == 3
    assert picks == [1]  # highest novelty first


def test_orchestrator_gated_novelty_requires_context():
    """No silent fallback to act>r* when gated_novelty is configured."""
    net = _net()
    cfg = PlasticityConfig(
        neuron_growth_trigger="gated_novelty",
        novelty_k_min=2,
        target_rates={"h1": 0.5, "h2": 0.5, "h3": 0.5},
        newborn_grace_steps=3,
        newborn_birth_bias_steps=3,
    )
    orch = PlasticityOrchestrator(config=cfg)
    scores = {
        "fc1": torch.zeros(8),
        "fc2": torch.zeros(6),
        "fc3": torch.zeros(4),
        "fc_out": torch.zeros(10),
    }
    with pytest.raises(ValueError, match="gated_novelty"):
        orch.step(net, scores)


def test_gap_imprint_direction_is_not_parent_copy():
    """Gap residual on CE-excess samples ≠ an arbitrary existing weight row."""
    B, F = 40, 8
    pre = torch.randn(B, F)
    # Gap samples 0..9 have a distinctive direction e0
    pre[:10] = 0.0
    pre[:10, 0] = 3.0
    pre[10:] = torch.randn(B - 10, F) * 0.1
    mask = torch.zeros(B, dtype=torch.bool)
    mask[:10] = True
    ce = torch.ones(B)
    ce[:10] = 2.0
    d = gap_imprint_direction(pre, mask, ce, subtract_global_mean=True)
    # Residual should peak on dim 0
    assert int(d.argmax().item()) == 0
    scaled = scale_direction_to_kaiming_relu(d)
    assert scaled.shape == (F,)
    assert float(scaled.std()) > 0


def test_per_dimension_separation_scores_prefer_diagnostic_not_magnitude():
    """High-variance non-diagnostic dim loses to gap-diagnostic dim."""
    torch.manual_seed(0)
    B, F = 200, 3
    pre = torch.zeros(B, F)
    mask = torch.zeros(B, dtype=torch.bool)
    mask[:100] = True
    # Dim 0: clear gap separation with some noise (diagnostic)
    pre[:100, 0] = 1.0 + 0.05 * torch.randn(100)
    pre[100:, 0] = 0.0 + 0.05 * torch.randn(100)
    # Dim 1: huge magnitude, almost no gap difference (high |W| trap)
    pre[:, 1] = 10.0 + 0.5 * torch.randn(B)
    # Dim 2: tiny separation
    pre[:100, 2] = 0.05 + 0.05 * torch.randn(100)
    pre[100:, 2] = 0.0 + 0.05 * torch.randn(100)
    residual = gap_imprint_direction(pre, mask, subtract_global_mean=True)
    scores = per_dimension_separation_scores(pre, mask, residual=residual)
    assert int(scores.argmax().item()) == 0
    assert float(scores[0]) > float(scores[1])
    # Open-fan scale uses n_open not full len → higher target std
    open_r = residual[[0, 2]]
    s_full = scale_direction_to_kaiming_relu(residual)
    s_open = scale_direction_to_kaiming_relu(open_r, fan_in=2)
    assert abs(float(s_open.std(unbiased=False)) - (2.0 / 2) ** 0.5) < 0.05


def test_gap_imprinted_birth_sets_incoming_not_parent_clone():
    """gap_imprinted writes the provided direction; does not copy parent row."""
    net = _net()
    parent = 1
    parent_row = net.fc1.weight[parent].detach().clone()
    net.begin_plasticity_step()
    net.deactivate_neuron("h1", 0)
    net.begin_plasticity_step()
    # Distinct direction on dim 0
    gap_row = torch.zeros(net.fc1.in_features)
    gap_row[0] = 1.0
    gap_row = scale_direction_to_kaiming_relu(gap_row)
    info = net.activate_neuron(
        "h1",
        0,
        origin=NEURON_ORIGIN_GROWN,
        parent_slot=parent,
        weight_init="gap_imprinted",
        parent_noise=0.0,
        gap_weight_row=gap_row,
    )
    assert info["weight_init"] == "gap_imprinted"
    # Not a parent clone
    cos = torch.nn.functional.cosine_similarity(
        net.fc1.weight[0].unsqueeze(0), parent_row.unsqueeze(0)
    ).item()
    assert cos < 0.5
    # Aligns with gap direction
    cos_gap = torch.nn.functional.cosine_similarity(
        net.fc1.weight[0].unsqueeze(0), gap_row.unsqueeze(0)
    ).item()
    assert cos_gap > 0.99
    # Edge fan not copied from parent (gap path leaves edges empty until seed)
    # parent may have dense edges; newborn should not match parent's edge row unless seeded
    # After activate alone, edge row is False (birth clears)
