"""Broadcast error encoder: per-sample residual packet + residual-energy modulator.

Encoder produces residual-only quantities for local importance signals:

- Path A (candidates 2–3): per-sample residual energy ``e_s^{(b)} = ||r_b||_2``
- Path B (candidate 4): full residual vectors ``r_per_sample``

Does **not** construct GEVB feedback matrices ``B_ell`` (owned by signals).

Batch-mean residual / energy is diagnostic only and must never be used as the
eligibility modulator — that collapses three-factor structure to a scalar times
mean co-activation. Use :meth:`BroadcastPacket.eligibility_energy_modulator`
(per-sample) instead of :func:`batch_mean_residual_as_eligibility_modulator`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class BroadcastPacket:
    """Global broadcast residual packet for one training step.

    Attributes
    ----------
    e_scalar_ce:
        Mean cross-entropy loss for the batch (diagnostic).
    e_vec_mean:
        Batch-mean residual vector ``mean_b r_b`` of shape ``[C]`` (diagnostic only).
    e_s_per_sample:
        Per-sample residual energy ``||r_b||_2`` of shape ``[B]`` — default
        scalar neuromodulator for candidates 2 and 3.
    r_per_sample:
        Full residuals ``r_b = softmax(z_b) - one_hot(y_b)`` of shape ``[B, C]``
        for candidate 4 (GEVB random feedback).
    step:
        Encoder step index (0-based, increments on each :meth:`BroadcastErrorEncoder.encode`).
    """

    e_scalar_ce: float
    e_vec_mean: Tensor  # [C]
    e_s_per_sample: Tensor  # [B]
    r_per_sample: Tensor  # [B, C]
    step: int

    def eligibility_energy_modulator(self) -> Tensor:
        """Per-sample residual energy for candidates 2–3 (shape ``[B]``).

        This is the only supported scalar eligibility modulator from this packet.
        Do not use ``||e_vec_mean||_2`` or a batch-scalar substitute.
        """
        return self.e_s_per_sample


def batch_mean_residual_as_eligibility_modulator(packet: BroadcastPacket) -> Tensor:
    """Reject misuse of batch-mean residual as an eligibility modulator.

    Batch-mean ``e_vec_mean`` (or ``||e_vec_mean||_2``) is logged for diagnostics
    only. Using it as a shared scalar with per-sample co-activations reduces the
    three-factor update to ``e_s * mean(a_i a_j)`` and nullifies sample alignment.

    Raises
    ------
    ValueError
        Always — callers must use :meth:`BroadcastPacket.eligibility_energy_modulator`
        / ``e_s_per_sample`` instead.
    """
    raise ValueError(
        "Batch-mean residual/energy must not be used as eligibility modulator; "
        "use BroadcastPacket.e_s_per_sample (per-sample residual energy ||r_b||_2) "
        "via eligibility_energy_modulator(). "
        f"(got packet step={packet.step}, e_vec_mean shape={tuple(packet.e_vec_mean.shape)})"
    )


class BroadcastErrorEncoder:
    """Map logits/targets/loss → :class:`BroadcastPacket` (residual only).

    Never constructs or stores GEVB feedback matrices ``B_ell`` or deltas.
    """

    def __init__(self) -> None:
        self._step: int = 0

    @property
    def step(self) -> int:
        """Next step index that will be written into the packet."""
        return self._step

    def reset_step(self, step: int = 0) -> None:
        """Reset the internal step counter (e.g. on new run / resume)."""
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")
        self._step = int(step)

    @torch.no_grad()
    def encode(self, logits: Tensor, targets: Tensor, loss: Tensor) -> BroadcastPacket:
        """Encode batch residuals and residual energy.

        Parameters
        ----------
        logits:
            Model logits of shape ``[B, C]``.
        targets:
            Class indices of shape ``[B]`` (long).
        loss:
            Scalar (or reducible) CE loss tensor; stored as ``float(loss)`` diagnostic.

        Returns
        -------
        BroadcastPacket
            Detached residual packet for this step.
        """
        if logits.ndim != 2:
            raise ValueError(f"logits must be [B, C], got shape {tuple(logits.shape)}")
        if targets.ndim != 1:
            raise ValueError(f"targets must be [B], got shape {tuple(targets.shape)}")
        if logits.shape[0] != targets.shape[0]:
            raise ValueError(
                f"batch size mismatch: logits B={logits.shape[0]} vs targets B={targets.shape[0]}"
            )

        batch_size, num_classes = logits.shape
        if targets.numel() and int(targets.max()) >= num_classes:
            raise ValueError(
                f"targets max={int(targets.max())} >= num_classes={num_classes}"
            )

        # r_b = softmax(z_b) - one_hot(y_b)
        probs = F.softmax(logits.detach(), dim=-1)
        one_hot = F.one_hot(targets.long(), num_classes=num_classes).to(
            dtype=probs.dtype, device=probs.device
        )
        r_per_sample = probs - one_hot  # [B, C]

        # e_s^{(b)} = ||r_b||_2
        e_s_per_sample = torch.linalg.vector_norm(r_per_sample, ord=2, dim=-1)  # [B]

        # Diagnostics only — not eligibility modulators
        e_vec_mean = r_per_sample.mean(dim=0)  # [C]
        e_scalar_ce = float(loss.detach()) if torch.is_tensor(loss) else float(loss)

        packet = BroadcastPacket(
            e_scalar_ce=e_scalar_ce,
            e_vec_mean=e_vec_mean,
            e_s_per_sample=e_s_per_sample,
            r_per_sample=r_per_sample,
            step=self._step,
        )
        self._step += 1
        return packet
