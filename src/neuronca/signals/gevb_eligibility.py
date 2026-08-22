"""GEVB-RF eligibility: fixed random ``B_ell`` maps residuals; scores are |EMA T_ij|.

Owns ``B_ell`` (not the broadcast encoder). Optional persist under ``meta/``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from neuronca.broadcast.error import BroadcastPacket
from neuronca.signals.base import (
    DEFAULT_DECAY,
    ConfigError,
    _BaseSignal,
    ema_update,
    get_pre_post,
    validate_layer_shapes,
)


def gevb_rng_seed(experiment_seed: int, gevb_feedback_seed: int = 0) -> int:
    """Deterministic seed for B_ell: hash(experiment.seed, gevb_feedback.seed, 'gevb_B')."""
    payload = f"{int(experiment_seed)}:{int(gevb_feedback_seed)}:gevb_B".encode()
    digest = hashlib.sha256(payload).hexdigest()
    # Fit into signed 32-bit range used by torch.Generator.manual_seed
    return int(digest[:16], 16) % (2**31)


def default_gevb_persist_path(experiment_seed: int, meta_dir: str | Path = "meta") -> Path:
    """``meta/gevb_B_seed{k}.pt`` path for run seed ``k``."""
    return Path(meta_dir) / f"gevb_B_seed{int(experiment_seed)}.pt"


class GEVBVectorEligibility(_BaseSignal):
    """GEVB / feedback-alignment style eligibility with fixed random B_ell."""

    name = "eligibility_gevb_rf"

    def __init__(
        self,
        decay: float = DEFAULT_DECAY,
        *,
        experiment_seed: int = 0,
        gevb_feedback_seed: int = 0,
        num_classes: int | None = None,
        layer_shapes: dict[str, tuple[int, int]] | None = None,
        persist_path: str | Path | None = None,
        init: str = "gaussian_scaled",
        init_scale: float = 1.0,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(decay=decay)
        if init != "gaussian_scaled":
            raise ValueError(f"unsupported gevb init {init!r}; only 'gaussian_scaled'")
        if float(init_scale) <= 0:
            raise ValueError(f"init_scale must be > 0; got {init_scale!r}")
        self.experiment_seed = int(experiment_seed)
        self.gevb_feedback_seed = int(gevb_feedback_seed)
        self.rng_seed = gevb_rng_seed(self.experiment_seed, self.gevb_feedback_seed)
        self.init_scale = float(init_scale)
        self._num_classes = num_classes
        self._init_layer_shapes = dict(layer_shapes) if layer_shapes is not None else None
        self.persist_path = Path(persist_path) if persist_path is not None else None
        self._device = torch.device(device) if device is not None else None

        self.B: dict[str, Tensor] = {}  # layer -> [n_out, C], requires_grad=False
        self._T_edge: dict[str, Tensor] = {}
        self._B_ready = False

        if self._init_layer_shapes is not None and self._num_classes is not None:
            self._build_or_load_B(
                self._init_layer_shapes,
                num_classes=self._num_classes,
                device=self._device or torch.device("cpu"),
            )

    def _on_reset(self) -> None:
        # Keep B_ell fixed across reset of eligibility traces (orchestrator constants).
        self._T_edge.clear()
        # Re-allocate T on next update via _on_allocate; B stays if already ready.
        # If B not ready yet, nothing to do.

    def _on_allocate(
        self,
        layer_shapes: Mapping[str, tuple[int, int]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        for layer, (n_out, n_in) in layer_shapes.items():
            self._T_edge[layer] = torch.zeros(n_out, n_in, device=device, dtype=dtype)

    def _validate_B_against_shapes(
        self,
        layer_shapes: Mapping[str, tuple[int, int]],
        *,
        num_classes: int,
    ) -> None:
        """Ensure in-memory B keys/shapes match the requested layer map."""
        b_keys = set(self.B.keys())
        req_keys = set(layer_shapes.keys())
        if b_keys != req_keys:
            raise ConfigError(
                f"eligibility_gevb_rf: B_ell keys {sorted(b_keys)} != "
                f"layer_shapes keys {sorted(req_keys)}"
            )
        for layer, (n_out, _n_in) in layer_shapes.items():
            expected = (n_out, num_classes)
            if self.B[layer].shape != expected:
                raise ConfigError(
                    f"eligibility_gevb_rf: B[{layer}] shape {tuple(self.B[layer].shape)} "
                    f"!= expected {expected}"
                )

    def _persist_seed_mismatch_message(self, payload: dict) -> str:
        stored_rng = payload.get("rng_seed")
        stored_exp = payload.get("experiment_seed")
        stored_fb = payload.get("gevb_feedback_seed")
        return (
            f"GEVB persist file {self.persist_path} seed mismatch: "
            f"file (experiment_seed={stored_exp}, gevb_feedback_seed={stored_fb}, "
            f"rng_seed={stored_rng}) vs instance (experiment_seed={self.experiment_seed}, "
            f"gevb_feedback_seed={self.gevb_feedback_seed}, rng_seed={self.rng_seed}). "
            f"Use a matching path or delete the stale file."
        )

    def _build_or_load_B(
        self,
        layer_shapes: Mapping[str, tuple[int, int]],
        *,
        num_classes: int,
        device: torch.device,
    ) -> None:
        if self._B_ready:
            self._validate_B_against_shapes(layer_shapes, num_classes=num_classes)
            # Move existing B to device if needed
            for k, mat in list(self.B.items()):
                if mat.device != device:
                    self.B[k] = mat.to(device=device)
            return

        loaded = False
        if self.persist_path is not None and self.persist_path.is_file():
            payload = torch.load(self.persist_path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or "B" not in payload:
                raise RuntimeError(
                    f"persist file {self.persist_path} missing 'B' dict payload"
                )
            # Reject silent reuse when feedback/experiment seeds differ
            stored_rng = payload.get("rng_seed")
            stored_exp = payload.get("experiment_seed")
            stored_fb = payload.get("gevb_feedback_seed")
            if stored_rng is not None and int(stored_rng) != int(self.rng_seed):
                raise RuntimeError(self._persist_seed_mismatch_message(payload))
            if stored_exp is not None and int(stored_exp) != int(self.experiment_seed):
                raise RuntimeError(self._persist_seed_mismatch_message(payload))
            if stored_fb is not None and int(stored_fb) != int(self.gevb_feedback_seed):
                raise RuntimeError(self._persist_seed_mismatch_message(payload))
            stored_scale = payload.get("init_scale")
            if stored_scale is not None and abs(float(stored_scale) - float(self.init_scale)) > 1e-12:
                raise RuntimeError(
                    f"persist file {self.persist_path} init_scale={stored_scale} != "
                    f"requested {self.init_scale}"
                )

            B_dict = payload["B"]
            self.B = {
                name: t.detach().to(device=device, dtype=torch.float32).requires_grad_(False)
                for name, t in B_dict.items()
            }
            self._validate_B_against_shapes(layer_shapes, num_classes=num_classes)
            loaded = True

        if not loaded:
            # Default gaussian_scaled: std = 1/sqrt(C); init_scale multiplies that.
            std = (1.0 / (float(num_classes) ** 0.5)) * float(self.init_scale)
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.rng_seed)
            self.B = {}
            # Draw in deterministic layer order for reproducibility
            for layer in sorted(layer_shapes.keys()):
                n_out, _n_in = layer_shapes[layer]
                mat = torch.randn(n_out, num_classes, generator=gen, dtype=torch.float32)
                mat = mat * std
                self.B[layer] = mat.to(device=device).requires_grad_(False)

            if self.persist_path is not None:
                self.persist_path.parent.mkdir(parents=True, exist_ok=True)
                cpu_B = {k: v.detach().cpu() for k, v in self.B.items()}
                torch.save(
                    {
                        "B": cpu_B,
                        "experiment_seed": self.experiment_seed,
                        "gevb_feedback_seed": self.gevb_feedback_seed,
                        "rng_seed": self.rng_seed,
                        "num_classes": num_classes,
                        "init_scale": float(self.init_scale),
                    },
                    self.persist_path,
                )

        # Freeze: never trainable
        for mat in self.B.values():
            mat.requires_grad_(False)
        self._num_classes = num_classes
        self._B_ready = True

    @property
    def feedback_matrices(self) -> dict[str, Tensor]:
        """Fixed random feedback B_ell (not trained)."""
        return self.B

    @torch.no_grad()
    def update(
        self,
        activations: dict[str, Tensor],
        layer_shapes: dict[str, tuple[int, int]],
        broadcast: BroadcastPacket,
        step: int,
    ) -> None:
        if not layer_shapes:
            raise ValueError("layer_shapes must be non-empty")
        validate_layer_shapes(layer_shapes, activations)

        sample = next(iter(activations.values()))
        r = broadcast.r_per_sample
        if r.ndim != 2:
            raise ValueError(f"r_per_sample must be [B, C], got {tuple(r.shape)}")
        B_batch, C = r.shape
        if B_batch != int(sample.shape[0]):
            raise ValueError(
                f"r_per_sample batch {B_batch} != activation batch {int(sample.shape[0])}"
            )
        if self._num_classes is not None and C != self._num_classes:
            raise ValueError(
                f"r_per_sample C={C} != configured num_classes={self._num_classes}"
            )

        device = sample.device
        r = r.detach().to(device=device, dtype=torch.float32)

        self._build_or_load_B(layer_shapes, num_classes=C, device=device)
        self._ensure_buffers(layer_shapes, device=device, dtype=torch.float32)
        decay = self.decay

        for layer, (n_out, n_in) in layer_shapes.items():
            a_pre, a_post = get_pre_post(activations, layer)
            # post activation width must match B_ell rows; a_post not used in GEVB edge
            # formula (only a_i pre and δ_j). Still validate shapes via layer_shapes.
            del a_post
            a_pre = a_pre.detach().to(dtype=torch.float32)
            B_ell = self.B[layer]  # [n_out, C]
            assert not B_ell.requires_grad
            assert B_ell.shape == (n_out, C)

            # δ_j^{(b)} = (B_ell r_b)_j  → [B, n_out]
            # r: [B, C], B_ell: [n_out, C] → r @ B_ell.T
            delta_j = r @ B_ell.transpose(0, 1)  # [B, n_out]

            # δ_ij = mean_b a_i δ_j → [n_out, n_in]
            # (delta_j^T @ a_pre) / B
            delta_edge = (delta_j.transpose(0, 1) @ a_pre) / float(B_batch)
            assert delta_edge.shape == (n_out, n_in)

            ema_update(self._T_edge[layer], delta_edge, decay)
            abs_T = self._T_edge[layer].abs()
            self._edge[layer].copy_(abs_T)
            self._neuron[layer].copy_(abs_T.mean(dim=1))  # mean abs incoming

        self._step = int(step)
