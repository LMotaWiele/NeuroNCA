"""Tests for local importance signal candidates.

Covers shapes, weight-zero invariance, no_grad during update, GEVB B_ell frozen,
and per-sample residual-energy guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from neuronca.broadcast import BroadcastErrorEncoder, BroadcastPacket
from neuronca.models import MLPReasoningNet
from neuronca.signals import (
    DEFAULT_DECAY,
    ActivationMagnitude,
    ConfigError,
    GEVBVectorEligibility,
    HebbianResidualEnergyCorr,
    LocalSignal,
    ScoreBundle,
    ThreeFactorScalarEligibility,
    build_signal,
)
from neuronca.signals.base import require_per_sample_modulator


# Tiny net for fast unit tests (still 3 hidden + out naming)
TINY_HIDDEN = (8, 6, 4)
TINY_IN = 12
TINY_C = 5
DECAY = 0.99


def _tiny_model(seed: int = 0) -> MLPReasoningNet:
    torch.manual_seed(seed)
    return MLPReasoningNet(input_dim=TINY_IN, hidden=TINY_HIDDEN, num_classes=TINY_C)


def _batch(seed: int = 1, B: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, TINY_IN, generator=g)
    y = torch.randint(0, TINY_C, (B,), generator=g)
    return x, y


def _forward_packet(
    model: MLPReasoningNet, x: torch.Tensor, y: torch.Tensor
) -> tuple[dict[str, torch.Tensor], BroadcastPacket, dict[str, tuple[int, int]]]:
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    packet = BroadcastErrorEncoder().encode(logits, y, loss)
    acts = {k: v.clone() for k, v in model.last_activations.items()}  # type: ignore[union-attr]
    shapes = model.layer_shapes()
    return acts, packet, shapes


def _all_candidates(
    *,
    decay: float = DECAY,
    experiment_seed: int = 0,
    persist_path: Path | None = None,
) -> list[LocalSignal]:
    return [
        ActivationMagnitude(decay=decay),
        HebbianResidualEnergyCorr(decay=decay),
        ThreeFactorScalarEligibility(decay=decay),
        GEVBVectorEligibility(
            decay=decay,
            experiment_seed=experiment_seed,
            gevb_feedback_seed=0,
            num_classes=TINY_C,
            persist_path=persist_path,
        ),
    ]


# ---------------------------------------------------------------------------
# Protocol / factory
# ---------------------------------------------------------------------------


def test_default_decay_is_0_99() -> None:
    assert DEFAULT_DECAY == 0.99


def test_build_signal_names() -> None:
    for name in (
        "activation",
        "hebbian_corr",
        "eligibility_residual_energy",
        "eligibility_gevb_rf",
    ):
        sig = build_signal(name, decay=DECAY, experiment_seed=0, num_classes=TINY_C)
        assert sig.name == name
        assert isinstance(sig, LocalSignal)


def test_build_signal_unknown() -> None:
    with pytest.raises(ValueError, match="unknown signal"):
        build_signal("not_a_signal")


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ActivationMagnitude(decay=DECAY),
        lambda: HebbianResidualEnergyCorr(decay=DECAY),
        lambda: ThreeFactorScalarEligibility(decay=DECAY),
        lambda: GEVBVectorEligibility(
            decay=DECAY, experiment_seed=0, num_classes=TINY_C
        ),
    ],
    ids=["activation", "hebbian_corr", "eligibility_residual_energy", "eligibility_gevb_rf"],
)
def test_score_shapes_match_layer_shapes(factory) -> None:
    model = _tiny_model()
    x, y = _batch()
    acts, packet, shapes = _forward_packet(model, x, y)

    sig = factory()
    sig.update(acts, shapes, packet, step=0)

    neurons = sig.neuron_scores()
    edges = sig.edge_scores()
    assert set(neurons.keys()) == set(shapes.keys())
    assert set(edges.keys()) == set(shapes.keys())

    for layer, (n_out, n_in) in shapes.items():
        assert neurons[layer].shape == (n_out,), layer
        assert edges[layer].shape == (n_out, n_in), layer
        assert neurons[layer].dtype == torch.float32
        assert edges[layer].dtype == torch.float32
        assert neurons[layer].device.type == "cpu"
        assert edges[layer].device.type == "cpu"


def test_snapshot_scores_is_deepcopy() -> None:
    model = _tiny_model()
    x, y = _batch()
    acts, packet, shapes = _forward_packet(model, x, y)

    sig = ActivationMagnitude(decay=DECAY)
    sig.update(acts, shapes, packet, step=3)
    snap = sig.snapshot_scores()
    assert isinstance(snap, ScoreBundle)
    assert snap.name == "activation"
    assert snap.step == 3

    # Mutate live scores via second update; snapshot must not change
    x2, y2 = _batch(seed=99, B=5)
    acts2, packet2, _ = _forward_packet(model, x2, y2)
    before = {k: v.clone() for k, v in snap.edge.items()}
    sig.update(acts2, shapes, packet2, step=4)
    for k, v in snap.edge.items():
        torch.testing.assert_close(v, before[k])


# ---------------------------------------------------------------------------
# Weight-zero invariance (no W leakage)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ActivationMagnitude(decay=DECAY),
        lambda: HebbianResidualEnergyCorr(decay=DECAY),
        lambda: ThreeFactorScalarEligibility(decay=DECAY),
        lambda: GEVBVectorEligibility(
            decay=DECAY, experiment_seed=42, gevb_feedback_seed=0, num_classes=TINY_C
        ),
    ],
    ids=["activation", "hebbian_corr", "eligibility_residual_energy", "eligibility_gevb_rf"],
)
def test_weight_zero_invariance(factory) -> None:
    """Zeroing model weights must not change scores if acts+broadcast are fixed."""
    model = _tiny_model(seed=7)
    x, y = _batch(seed=11)
    acts, packet, shapes = _forward_packet(model, x, y)

    sig_a = factory()
    sig_a.update(acts, shapes, packet, step=0)
    snap_a = sig_a.snapshot_scores()

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

    # Re-use the same frozen activations and broadcast (do not re-forward)
    sig_b = factory()
    sig_b.update(acts, shapes, packet, step=0)
    snap_b = sig_b.snapshot_scores()

    for layer in shapes:
        torch.testing.assert_close(snap_a.neuron[layer], snap_b.neuron[layer])
        torch.testing.assert_close(snap_a.edge[layer], snap_b.edge[layer])


# ---------------------------------------------------------------------------
# no_grad during update
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ActivationMagnitude(decay=DECAY),
        lambda: HebbianResidualEnergyCorr(decay=DECAY),
        lambda: ThreeFactorScalarEligibility(decay=DECAY),
        lambda: GEVBVectorEligibility(
            decay=DECAY, experiment_seed=0, num_classes=TINY_C
        ),
    ],
    ids=["activation", "hebbian_corr", "eligibility_residual_energy", "eligibility_gevb_rf"],
)
def test_update_under_no_grad_scores_detached(factory) -> None:
    model = _tiny_model()
    x, y = _batch()
    # Graph-bearing activations
    x_req = x.clone().requires_grad_(True)
    logits = model(x_req)
    loss = F.cross_entropy(logits, y)
    packet = BroadcastErrorEncoder().encode(logits, y, loss)
    # Live activations with grad
    acts = {
        "act_in": x_req,
        "act_fc1": model.last_activations["act_fc1"].clone().requires_grad_(True),  # type: ignore[index]
        "act_fc2": model.last_activations["act_fc2"].clone().requires_grad_(True),  # type: ignore[index]
        "act_fc3": model.last_activations["act_fc3"].clone().requires_grad_(True),  # type: ignore[index]
        "act_fc_out": logits,
    }
    shapes = model.layer_shapes()

    sig = factory()
    # Grad mode enabled — update itself must still not retain graph on scores
    assert torch.is_grad_enabled()
    sig.update(acts, shapes, packet, step=0)

    for t in list(sig.neuron_scores().values()) + list(sig.edge_scores().values()):
        assert not t.requires_grad
        assert t.grad_fn is None


# ---------------------------------------------------------------------------
# GEVB B_ell not trained + persistence
# ---------------------------------------------------------------------------


def test_gevb_B_ell_not_trained() -> None:
    model = _tiny_model()
    x, y = _batch()
    acts, packet, shapes = _forward_packet(model, x, y)

    sig = GEVBVectorEligibility(
        decay=DECAY, experiment_seed=0, gevb_feedback_seed=1, num_classes=TINY_C
    )
    sig.update(acts, shapes, packet, step=0)

    assert sig.B
    for layer, mat in sig.B.items():
        assert not mat.requires_grad, layer
        n_out, _ = shapes[layer]
        assert mat.shape == (n_out, TINY_C)


def test_gevb_B_ell_persist_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "gevb_B_seed0.pt"
    model = _tiny_model()
    x, y = _batch()
    acts, packet, shapes = _forward_packet(model, x, y)

    sig1 = GEVBVectorEligibility(
        decay=DECAY,
        experiment_seed=0,
        gevb_feedback_seed=3,
        num_classes=TINY_C,
        layer_shapes=shapes,
        persist_path=path,
    )
    assert path.is_file()
    B1 = {k: v.clone() for k, v in sig1.B.items()}

    sig2 = GEVBVectorEligibility(
        decay=DECAY,
        experiment_seed=0,
        gevb_feedback_seed=3,
        num_classes=TINY_C,
        layer_shapes=shapes,
        persist_path=path,
    )
    for layer in shapes:
        torch.testing.assert_close(sig2.B[layer], B1[layer])

    # Same B after update path as well
    sig1.update(acts, shapes, packet, step=0)
    sig2.update(acts, shapes, packet, step=0)
    for layer in shapes:
        torch.testing.assert_close(sig1.edge_scores()[layer], sig2.edge_scores()[layer])


def test_gevb_seed_deterministic() -> None:
    shapes = {
        "fc1": (8, 12),
        "fc2": (6, 8),
        "fc3": (4, 6),
        "fc_out": (5, 4),
    }
    a = GEVBVectorEligibility(
        experiment_seed=1, gevb_feedback_seed=2, num_classes=5, layer_shapes=shapes
    )
    b = GEVBVectorEligibility(
        experiment_seed=1, gevb_feedback_seed=2, num_classes=5, layer_shapes=shapes
    )
    c = GEVBVectorEligibility(
        experiment_seed=1, gevb_feedback_seed=9, num_classes=5, layer_shapes=shapes
    )
    for layer in shapes:
        torch.testing.assert_close(a.B[layer], b.B[layer])
        assert not torch.equal(a.B[layer], c.B[layer])


def test_gevb_persist_seed_mismatch_raises(tmp_path: Path) -> None:
    """Loading B from a path written with a different gevb_feedback_seed must fail."""
    path = tmp_path / "gevb_B_seed0.pt"
    shapes = {
        "fc1": (8, 12),
        "fc2": (6, 8),
        "fc3": (4, 6),
        "fc_out": (5, 4),
    }
    GEVBVectorEligibility(
        experiment_seed=0,
        gevb_feedback_seed=0,
        num_classes=5,
        layer_shapes=shapes,
        persist_path=path,
    )
    assert path.is_file()
    with pytest.raises(RuntimeError, match="seed mismatch"):
        GEVBVectorEligibility(
            experiment_seed=0,
            gevb_feedback_seed=999,
            num_classes=5,
            layer_shapes=shapes,
            persist_path=path,
        )


def test_layer_shapes_keyset_change_raises() -> None:
    """Partial then full layer_shapes after allocation raises ConfigError."""
    model = _tiny_model()
    x, y = _batch()
    acts, packet, shapes = _forward_packet(model, x, y)
    partial = {"fc1": shapes["fc1"]}

    sig = ActivationMagnitude(decay=DECAY)
    sig.update(acts, partial, packet, step=0)
    with pytest.raises(ConfigError, match="layer_shapes keys changed"):
        sig.update(acts, shapes, packet, step=1)


# ---------------------------------------------------------------------------
# Per-sample residual energy guard
# ---------------------------------------------------------------------------


def test_require_per_sample_rejects_scalar() -> None:
    with pytest.raises(ConfigError, match="batch-scalar"):
        require_per_sample_modulator(torch.tensor(1.5), batch_size=8, context="test")


def test_require_per_sample_rejects_numel1_for_batch() -> None:
    with pytest.raises(ConfigError, match="batch-scalar|numel"):
        require_per_sample_modulator(torch.tensor([1.5]), batch_size=8, context="test")


def test_eligibility_rejects_batch_scalar_modulator() -> None:
    """Scalar / length-1 e_s with per-sample a must raise ConfigError on update."""
    model = _tiny_model()
    x, y = _batch(B=6)
    acts, packet, shapes = _forward_packet(model, x, y)

    bad_0dim = BroadcastPacket(
        e_scalar_ce=packet.e_scalar_ce,
        e_vec_mean=packet.e_vec_mean,
        e_s_per_sample=torch.tensor(0.5),  # 0-dim scalar
        r_per_sample=packet.r_per_sample,
        step=packet.step,
    )
    bad_mean = BroadcastPacket(
        e_scalar_ce=packet.e_scalar_ce,
        e_vec_mean=packet.e_vec_mean,
        e_s_per_sample=packet.e_s_per_sample.mean(),  # 0-dim via mean
        r_per_sample=packet.r_per_sample,
        step=packet.step,
    )
    bad_len1 = BroadcastPacket(
        e_scalar_ce=packet.e_scalar_ce,
        e_vec_mean=packet.e_vec_mean,
        e_s_per_sample=torch.tensor([0.5]),  # [1] vs B>1
        r_per_sample=packet.r_per_sample,
        step=packet.step,
    )

    for bad_packet in (bad_0dim, bad_mean, bad_len1):
        for sig in (
            HebbianResidualEnergyCorr(decay=DECAY),
            ThreeFactorScalarEligibility(decay=DECAY),
        ):
            with pytest.raises(ConfigError):
                sig.update(acts, shapes, bad_packet, step=0)


def test_eligibility_formula_uses_per_sample_e() -> None:
    """δ = mean_b a_i a_j e_s^b differs from e_mean * mean(a_i a_j) when e varies."""
    torch.manual_seed(0)
    B, n_in, n_out = 4, 3, 2
    a_pre = torch.randn(B, n_in)
    a_post = torch.randn(B, n_out)
    e_s = torch.tensor([0.1, 1.0, 2.0, 0.5])

    weighted = a_post * e_s[:, None]
    delta_per = (weighted.transpose(0, 1) @ a_pre) / float(B)
    delta_collapsed = e_s.mean() * (a_post.transpose(0, 1) @ a_pre) / float(B)
    # With varying e_s these should not match
    assert not torch.allclose(delta_per, delta_collapsed, atol=1e-5)

    # Wire through signal with synthetic acts matching tiny naming via manual update path
    # Use full model path with controlled broadcast
    model = _tiny_model()
    shapes = model.layer_shapes()
    # Synthetic activations with known batch
    acts = {
        "act_in": torch.randn(B, TINY_IN),
        "act_fc1": torch.randn(B, TINY_HIDDEN[0]),
        "act_fc2": torch.randn(B, TINY_HIDDEN[1]),
        "act_fc3": torch.randn(B, TINY_HIDDEN[2]),
        "act_fc_out": torch.randn(B, TINY_C),
    }
    e_var = torch.tensor([0.1, 2.0, 0.5, 1.5])
    r = torch.randn(B, TINY_C)
    packet = BroadcastPacket(
        e_scalar_ce=1.0,
        e_vec_mean=r.mean(0),
        e_s_per_sample=e_var,
        r_per_sample=r,
        step=0,
    )
    # From zero init: T = (1-λ)δ after first update; s = |T|
    sig = ThreeFactorScalarEligibility(decay=0.5)
    sig.update(acts, shapes, packet, step=0)

    # Manual δ for fc1
    a_pre = acts["act_in"]
    a_post = acts["act_fc1"]
    weighted = a_post * e_var[:, None]
    delta = (weighted.transpose(0, 1) @ a_pre) / float(B)
    expected_edge = ((1.0 - 0.5) * delta).abs()  # from zeros
    torch.testing.assert_close(sig.edge_scores()["fc1"], expected_edge, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# EMA / multi-step sanity
# ---------------------------------------------------------------------------


def test_activation_ema_update() -> None:
    model = _tiny_model()
    shapes = model.layer_shapes()
    B = 4
    acts1 = {
        "act_in": torch.ones(B, TINY_IN),
        "act_fc1": torch.ones(B, TINY_HIDDEN[0]),
        "act_fc2": torch.ones(B, TINY_HIDDEN[1]),
        "act_fc3": torch.ones(B, TINY_HIDDEN[2]),
        "act_fc_out": torch.ones(B, TINY_C),
    }
    packet = BroadcastPacket(
        e_scalar_ce=0.0,
        e_vec_mean=torch.zeros(TINY_C),
        e_s_per_sample=torch.ones(B),
        r_per_sample=torch.zeros(B, TINY_C),
        step=0,
    )
    lam = 0.99
    sig = ActivationMagnitude(decay=lam)
    sig.update(acts1, shapes, packet, step=0)
    # From zero: s = (1-λ) * 1
    expected = 1.0 - lam
    torch.testing.assert_close(
        sig.neuron_scores()["fc1"],
        torch.full((TINY_HIDDEN[0],), expected),
        atol=1e-6,
        rtol=1e-5,
    )
    # Edge |a_i||a_j| = 1
    torch.testing.assert_close(
        sig.edge_scores()["fc1"],
        torch.full((TINY_HIDDEN[0], TINY_IN), expected),
        atol=1e-6,
        rtol=1e-5,
    )


def test_reset_clears_traces() -> None:
    model = _tiny_model()
    x, y = _batch()
    acts, packet, shapes = _forward_packet(model, x, y)
    sig = ActivationMagnitude(decay=DECAY)
    sig.update(acts, shapes, packet, step=0)
    assert any(v.abs().sum() > 0 for v in sig.neuron_scores().values())
    sig.reset()
    # After reset, scores empty until next update
    assert sig.neuron_scores() == {}
    assert sig.edge_scores() == {}


def test_candidates_are_local_signal_protocol() -> None:
    for sig in _all_candidates():
        assert isinstance(sig, LocalSignal)
        assert hasattr(sig, "name")
