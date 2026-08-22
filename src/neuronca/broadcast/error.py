"""Broadcast residual packet for local signals (per-sample energy / vectors).

Does not own GEVB ``B_ell``. Batch-mean residual is diagnostic only — never
the eligibility modulator (use ``e_s_per_sample``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class BroadcastPacket:
    """One-step residual packet: per-sample energy ``[B]``, residuals ``[B, C]``.

    ``e_vec_mean`` is diagnostic only; eligibility uses ``e_s_per_sample``.
    """

    e_scalar_ce: float
    e_vec_mean: Tensor  # [C]
    e_s_per_sample: Tensor  # [B]
    r_per_sample: Tensor  # [B, C]
    step: int

    def eligibility_energy_modulator(self) -> Tensor:
        """Per-sample residual energy (the only supported eligibility modulator)."""
        return self.e_s_per_sample


def batch_mean_residual_as_eligibility_modulator(packet: BroadcastPacket) -> Tensor:
    """Always raises: batch-mean residual is not an eligibility modulator."""
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
