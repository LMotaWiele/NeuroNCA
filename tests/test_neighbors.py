"""Tests for grid vs feature neighbor graphs (Experiment 1)."""

from __future__ import annotations

import torch

from neuronca.models.dynamic_mlp import DynamicMLPReasoningNet
from neuronca.plasticity.neighbors import (
    NeighborGraph,
    cosine_similarity_matrix,
    mean_neighbor_count,
    neighbor_mask_feature_topk,
    neighbor_mask_within_radius,
    neighbor_set_churn,
    probe_activation_signatures,
)
from neuronca.plasticity.rules import gated_novelty_scores_for_layer


def test_feature_topk_size_matches_k():
    H, B = 8, 16
    active = torch.ones(H, dtype=torch.bool)
    # Distinct signatures so top-k is well-defined
    acts = torch.randn(B, H)
    sig = probe_activation_signatures(acts, active)
    k = 3
    mask = neighbor_mask_feature_topk(sig, active, k)
    counts = mask.sum(dim=1)
    assert int(counts.min()) == k
    assert int(counts.max()) == k
    assert not bool(mask.diag().any())


def test_matched_k_feature_matches_grid_mean():
    net = DynamicMLPReasoningNet(
        hidden_init=(32, 16, 8),
        hidden_max=(64, 32, 16),
        coord_seed=0,
    )
    B = 24
    acts = {
        h: torch.randn(B, net.coords(h).shape[0]) for h in ("h1", "h2", "h3")
    }
    for h, a in acts.items():
        a[:, ~net.active_mask(h)] = 0
    g = NeighborGraph(metric="feature", radius=0.35)
    r = g.refresh(net, layer_acts=acts)
    for h in ("h1", "h2", "h3"):
        grid_mean = r.grid_mean_count_by_layer[h]
        k = r.matched_k_by_layer[h]
        assert k == max(int(round(grid_mean)), 1)
        # Feature mean count should be ~k (exactly k for all full-degree units)
        assert abs(r.mean_count_by_layer[h] - k) < 0.5


def test_churn_detects_turnover():
    H = 5
    active = torch.ones(H, dtype=torch.bool)
    prev = torch.zeros(H, H, dtype=torch.bool)
    prev[0, 1] = True
    prev[0, 2] = True
    new = torch.zeros(H, H, dtype=torch.bool)
    new[0, 1] = True
    new[0, 3] = True  # lost 2, kept 1 → lost fraction 0.5 for unit 0
    # Only unit 0 has neighbors in prev
    ch = neighbor_set_churn(prev, new, active)
    assert abs(ch - 0.5) < 1e-6


def test_gated_novelty_accepts_feature_mask():
    B, H = 32, 6
    coords = torch.zeros(H, 2)
    for i in range(H):
        coords[i, 0] = 0.1 * (i % 3)
        coords[i, 1] = 0.1 * (i // 3)
    active = torch.ones(H, dtype=torch.bool)
    acts = torch.ones(B, H) * 0.5
    ce = torch.ones(B) * 0.5
    acts[:16] = 0.05
    ce[:16] = 2.0
    # Feature mask = full connectivity among active (like dense local)
    mask = torch.ones(H, H, dtype=torch.bool)
    mask.fill_diagonal_(False)
    stats, cands, masks = gated_novelty_scores_for_layer(
        acts, ce, coords, active, radius=0.5, k_min=3, neighbor_mask=mask
    )
    assert stats.n_under_floor == 0
    assert stats.n_density_pass == H
    assert len(cands) >= 1


def test_cosine_self_excluded():
    H, B = 4, 10
    active = torch.ones(H, dtype=torch.bool)
    acts = torch.randn(B, H)
    sig = probe_activation_signatures(acts, active)
    sim = cosine_similarity_matrix(sig, active)
    assert float(sim.diag().max()) < -1e8
