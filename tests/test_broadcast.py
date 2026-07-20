"""Tests for broadcast residual packet and energy modulator."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from neuronca.broadcast import (
    BroadcastErrorEncoder,
    BroadcastPacket,
    batch_mean_residual_as_eligibility_modulator,
)
from neuronca.broadcast.error import BroadcastErrorEncoder as EncoderFromModule


def _manual_residual(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Reference r_b = softmax(z_b) - one_hot(y_b)."""
    probs = F.softmax(logits, dim=-1)
    one_hot = F.one_hot(targets.long(), num_classes=logits.shape[-1]).to(probs.dtype)
    return probs - one_hot


@pytest.mark.parametrize("batch_size,num_classes", [(1, 3), (4, 10), (8, 5), (32, 10)])
def test_packet_shapes(batch_size: int, num_classes: int) -> None:
    """BroadcastPacket fields have shapes [B], [B, C], [C] as designed."""
    torch.manual_seed(0)
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    loss = F.cross_entropy(logits, targets)

    enc = BroadcastErrorEncoder()
    packet = enc.encode(logits, targets, loss)

    assert packet.e_s_per_sample.shape == (batch_size,)
    assert packet.r_per_sample.shape == (batch_size, num_classes)
    assert packet.e_vec_mean.shape == (num_classes,)
    assert isinstance(packet.e_scalar_ce, float)
    assert packet.step == 0


def test_e_s_per_sample_equals_row_l2_norm() -> None:
    """e_s_per_sample equals ||r_b||_2 per row."""
    torch.manual_seed(1)
    B, C = 16, 10
    logits = torch.randn(B, C)
    targets = torch.randint(0, C, (B,))
    loss = F.cross_entropy(logits, targets)

    packet = BroadcastErrorEncoder().encode(logits, targets, loss)
    expected = torch.linalg.vector_norm(packet.r_per_sample, ord=2, dim=-1)
    torch.testing.assert_close(packet.e_s_per_sample, expected)
    # Nonnegative energy
    assert (packet.e_s_per_sample >= 0).all()


def test_residual_matches_softmax_minus_one_hot() -> None:
    """r_per_sample matches softmax(logits) - one_hot(targets)."""
    torch.manual_seed(2)
    B, C = 7, 4
    logits = torch.randn(B, C)
    targets = torch.randint(0, C, (B,))
    loss = torch.tensor(0.5)

    packet = BroadcastErrorEncoder().encode(logits, targets, loss)
    ref = _manual_residual(logits, targets)
    torch.testing.assert_close(packet.r_per_sample, ref)


def test_residual_sums_to_zero_over_classes() -> None:
    """sum_c r_bc = 0 because softmax and one_hot both sum to 1."""
    torch.manual_seed(3)
    B, C = 12, 6
    logits = torch.randn(B, C)
    targets = torch.randint(0, C, (B,))
    packet = BroadcastErrorEncoder().encode(logits, targets, F.cross_entropy(logits, targets))

    row_sums = packet.r_per_sample.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.zeros(B), atol=1e-6, rtol=1e-5)


def test_e_vec_mean_is_batch_mean_of_residuals() -> None:
    """e_vec_mean is diagnostic batch-mean residual (not the modulator)."""
    torch.manual_seed(4)
    B, C = 9, 5
    logits = torch.randn(B, C)
    targets = torch.randint(0, C, (B,))
    packet = BroadcastErrorEncoder().encode(logits, targets, torch.tensor(1.0))

    torch.testing.assert_close(packet.e_vec_mean, packet.r_per_sample.mean(dim=0))


def test_perfect_prediction_near_zero_residual() -> None:
    """Confident correct logits yield near-zero residual energy."""
    C = 5
    # Large positive on the true class → softmax ≈ one_hot
    logits = torch.full((3, C), -20.0)
    targets = torch.tensor([0, 2, 4])
    for b, y in enumerate(targets.tolist()):
        logits[b, y] = 20.0
    packet = BroadcastErrorEncoder().encode(logits, targets, torch.tensor(0.0))
    assert (packet.e_s_per_sample < 1e-6).all()
    torch.testing.assert_close(
        packet.r_per_sample, torch.zeros_like(packet.r_per_sample), atol=1e-6, rtol=0.0
    )


def test_e_scalar_ce_from_loss() -> None:
    """e_scalar_ce stores the provided loss as a float diagnostic."""
    logits = torch.randn(4, 3)
    targets = torch.randint(0, 3, (4,))
    loss = F.cross_entropy(logits, targets)
    packet = BroadcastErrorEncoder().encode(logits, targets, loss)
    assert packet.e_scalar_ce == pytest.approx(float(loss.detach()))


def test_step_increments() -> None:
    """Encoder step counter increments on each encode."""
    enc = BroadcastErrorEncoder()
    assert enc.step == 0
    logits = torch.randn(2, 3)
    targets = torch.tensor([0, 1])
    loss = torch.tensor(0.1)
    p0 = enc.encode(logits, targets, loss)
    p1 = enc.encode(logits, targets, loss)
    assert p0.step == 0
    assert p1.step == 1
    assert enc.step == 2
    enc.reset_step(10)
    p2 = enc.encode(logits, targets, loss)
    assert p2.step == 10


def test_eligibility_energy_modulator_is_per_sample() -> None:
    """Supported modulator is e_s_per_sample, not batch-mean energy."""
    torch.manual_seed(5)
    B, C = 5, 4
    logits = torch.randn(B, C)
    targets = torch.randint(0, C, (B,))
    packet = BroadcastErrorEncoder().encode(logits, targets, torch.tensor(1.0))

    mod = packet.eligibility_energy_modulator()
    torch.testing.assert_close(mod, packet.e_s_per_sample)
    assert mod.shape == (B,)
    # Batch-mean energy is a different scalar and must not equal the per-sample vector
    batch_mean_energy = torch.linalg.vector_norm(packet.e_vec_mean, ord=2)
    assert batch_mean_energy.ndim == 0
    assert not torch.allclose(mod, batch_mean_energy.expand_as(mod))


def test_batch_mean_modulator_misuse_raises() -> None:
    """Calling the batch-mean eligibility helper always raises (document misuse).

    Batch-mean residual must not be used as eligibility modulator; use
    per-sample e_s_per_sample instead.
    """
    logits = torch.randn(3, 4)
    targets = torch.tensor([0, 1, 2])
    packet = BroadcastErrorEncoder().encode(logits, targets, torch.tensor(0.5))

    with pytest.raises(ValueError, match="must not be used as eligibility modulator"):
        batch_mean_residual_as_eligibility_modulator(packet)


def test_encoder_does_not_attach_gevb_matrices() -> None:
    """Encoder is residual-only: no B_ell / GEVB feedback attributes on packet."""
    packet = BroadcastErrorEncoder().encode(
        torch.randn(2, 3), torch.tensor([0, 1]), torch.tensor(0.2)
    )
    assert not hasattr(packet, "B")
    assert not hasattr(packet, "B_ell")
    assert not hasattr(packet, "delta")
    assert not hasattr(packet, "gevb_feedback")
    # Packet fields are only the residual interface
    fields = {f.name for f in packet.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    assert fields == {
        "e_scalar_ce",
        "e_vec_mean",
        "e_s_per_sample",
        "r_per_sample",
        "step",
    }


def test_encode_detaches_from_logits_graph() -> None:
    """Residuals do not require grad even if logits did."""
    logits = torch.randn(4, 3, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 0])
    loss = F.cross_entropy(logits, targets)
    packet = BroadcastErrorEncoder().encode(logits, targets, loss)
    assert not packet.r_per_sample.requires_grad
    assert not packet.e_s_per_sample.requires_grad
    assert not packet.e_vec_mean.requires_grad


def test_invalid_shapes_raise() -> None:
    enc = BroadcastErrorEncoder()
    with pytest.raises(ValueError, match=r"\[B, C\]"):
        enc.encode(torch.randn(3), torch.tensor([0, 1, 2]), torch.tensor(0.0))
    with pytest.raises(ValueError, match="batch size mismatch"):
        enc.encode(torch.randn(2, 3), torch.tensor([0, 1, 2]), torch.tensor(0.0))


def test_public_exports() -> None:
    """Package exports match design interface."""
    assert EncoderFromModule is BroadcastErrorEncoder
    assert callable(batch_mean_residual_as_eligibility_modulator)
    assert BroadcastPacket.__dataclass_fields__  # dataclass
