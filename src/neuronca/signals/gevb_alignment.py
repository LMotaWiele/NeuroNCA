"""FA alignment diagnostics for GEVB-RF: is B_ℓ still unaligned with ∂L/∂z?

Primary: probe-mean angle between ``B r`` and ``∂L/∂z_ℓ``. Secondary: principal
angle of ``col(B)`` vs the top-k left-singular subspace of ``W`` (raw Frobenius
W-vs-B is ill-posed). Low late Spearman is uninterpretable while FA is stalled.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from neuronca.broadcast.error import BroadcastErrorEncoder
from neuronca.signals.gevb_eligibility import GEVBVectorEligibility
from neuronca.training.loop import _unpack_batch, resolve_device


def _angle_deg(u: Tensor, v: Tensor, *, eps: float = 1e-12) -> float:
    """Geometric angle in degrees between two vectors (flattened).

    Returns a value in ``[0, 180]`` (signed direction matters for FA credit),
    or nan if either vector is ~0. Closer to 0° is better; ~180° is
    anti-aligned (wrong sign).
    """
    uf = u.detach().float().reshape(-1)
    vf = v.detach().float().reshape(-1)
    nu = torch.linalg.vector_norm(uf)
    nv = torch.linalg.vector_norm(vf)
    if float(nu) < eps or float(nv) < eps:
        return float("nan")
    cos = float(torch.dot(uf, vf) / (nu * nv))
    cos = max(-1.0, min(1.0, cos))
    return float(math.degrees(math.acos(cos)))


def subspace_principal_angle_deg(
    W: Tensor,
    B: Tensor,
    *,
    eps: float = 1e-8,
) -> float:
    """Smallest principal angle (degrees) between col(B) and top-k left SVs of W.

    Both matrices must share the leading dimension ``n_out``. Uses the top
    ``k = min(B.shape[1], rank(W), n_out)`` left singular vectors of ``W`` as
    the forward subspace (see module docstring for why full ``col(W)`` is
    vacuous on wide layers).
    """
    if W.ndim != 2 or B.ndim != 2:
        raise ValueError("W and B must be 2-D")
    if W.shape[0] != B.shape[0]:
        raise ValueError(
            f"shared postsynaptic dim mismatch: W {tuple(W.shape)} vs B {tuple(B.shape)}"
        )
    Wf = W.detach().float().cpu()
    Bf = B.detach().float().cpu()

    # Top-k left singular vectors of W
    try:
        U, S, _Vh = torch.linalg.svd(Wf, full_matrices=False)
    except RuntimeError:
        return float("nan")
    if S.numel() == 0:
        return float("nan")
    rank_w = int((S > eps * S.max().clamp_min(eps)).sum().item())
    rank_w = max(rank_w, 1)
    k = min(int(Bf.shape[1]), rank_w, int(U.shape[1]))
    if k <= 0:
        return float("nan")
    Qw = U[:, :k]

    # Orthonormal basis for col(B)
    Qb, Rb = torch.linalg.qr(Bf, mode="reduced")
    diag = torch.abs(torch.diag(Rb)) if Rb.numel() else torch.tensor([])
    if diag.numel() == 0:
        return float("nan")
    rank_b = int((diag > eps * diag.max().clamp_min(eps)).sum().item())
    rank_b = max(rank_b, 1)
    Qb = Qb[:, :rank_b]

    M = Qw.T @ Qb
    try:
        s = torch.linalg.svdvals(M)
    except RuntimeError:
        return float("nan")
    c = float(s.max().clamp(-1.0, 1.0))
    return float(math.degrees(math.acos(c)))


@torch.no_grad()
def _mean_teaching_angle_deg(
    delta_fa: Tensor,
    delta_true: Tensor,
) -> float:
    """Mean per-sample angle between FA and true δ; both ``[B, n_out]``."""
    if delta_fa.shape != delta_true.shape:
        raise ValueError(
            f"shape mismatch FA {tuple(delta_fa.shape)} vs true {tuple(delta_true.shape)}"
        )
    angles = []
    for b in range(delta_fa.shape[0]):
        ang = _angle_deg(delta_fa[b], delta_true[b])
        if math.isfinite(ang):
            angles.append(ang)
    if not angles:
        return float("nan")
    return float(sum(angles) / len(angles))


def compute_gevb_alignment(
    model: nn.Module,
    gevb: GEVBVectorEligibility,
    probe_loader: DataLoader,
    *,
    criterion: nn.Module | None = None,
    encoder: BroadcastErrorEncoder | None = None,
    device: str | torch.device | None = None,
    max_batches: int = 4,
    layer_names: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute per-layer GEVB alignment diagnostics on the probe set.

    Returns
    -------
    dict
        ``layer → {fa_signal_angle_deg, subspace_principal_angle_deg, n_samples}``
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if encoder is None:
        encoder = BroadcastErrorEncoder()
    device_t = resolve_device(device)
    model = model.to(device_t)
    model.eval()

    linear_mods: dict[str, nn.Linear] = {
        n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)
    }
    if hasattr(model, "layer_shapes") and callable(model.layer_shapes):
        layer_shapes = dict(model.layer_shapes())
    else:
        layer_shapes = {
            n: (m.out_features, m.in_features) for n, m in linear_mods.items()
        }

    # Ensure B_ℓ is materialised (does not touch eligibility traces).
    if not gevb.B:
        num_classes = int(gevb._num_classes or 10)  # noqa: SLF001
        # Infer C from a one-batch encode if needed
        sample0 = next(iter(probe_loader))
        x0, y0 = _unpack_batch(sample0)
        x0 = x0.to(device_t)
        y0 = y0.to(device_t)
        with torch.no_grad():
            logits0 = model(x0)
            loss0 = criterion(logits0, y0)
            packet0 = encoder.encode(logits0, y0, loss0)
            num_classes = int(packet0.r_per_sample.shape[1])
        gevb._build_or_load_B(  # noqa: SLF001 — diagnostic access to fixed B
            layer_shapes,
            num_classes=num_classes,
            device=device_t,
        )

    target_layers = list(layer_names) if layer_names is not None else sorted(
        set(linear_mods.keys()) & set(gevb.B.keys())
    )
    if not target_layers:
        return {}

    # Accumulators
    fa_sums: dict[str, float] = {ℓ: 0.0 for ℓ in target_layers}
    fa_counts: dict[str, int] = {ℓ: 0 for ℓ in target_layers}
    n_samples = 0
    batches = 0

    for batch in probe_loader:
        if batches >= max_batches:
            break
        x, y = _unpack_batch(batch)
        x = x.to(device_t)
        y = y.to(device_t)
        if y.ndim > 1 and not y.is_floating_point():
            y = y.view(-1)

        # Capture linear outputs for true δ = ∂L/∂z
        linear_out: dict[str, Tensor] = {}

        def _make_hook(name: str):
            def hook(_mod: nn.Module, _inp: tuple, out: Tensor) -> None:
                # retain grad for non-leaf
                out.retain_grad()
                linear_out[name] = out

            return hook

        handles = []
        for name in target_layers:
            if name in linear_mods:
                handles.append(linear_mods[name].register_forward_hook(_make_hook(name)))

        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        # Broadcast residual for FA teaching signal (detached)
        with torch.no_grad():
            packet = encoder.encode(logits.detach(), y, loss.detach())
            r = packet.r_per_sample.to(device=device_t, dtype=torch.float32)

        loss.backward()

        for name in target_layers:
            if name not in linear_out or name not in gevb.B:
                continue
            z = linear_out[name]
            if z.grad is None:
                continue
            delta_true = z.grad.detach()  # [B, n_out]
            B_ell = gevb.B[name].to(device=device_t, dtype=torch.float32)
            # δ_fa = r @ B^T → [B, n_out]
            delta_fa = r @ B_ell.transpose(0, 1)
            ang = _mean_teaching_angle_deg(delta_fa, delta_true)
            if math.isfinite(ang):
                fa_sums[name] += ang * int(z.shape[0])
                fa_counts[name] += int(z.shape[0])

        for h in handles:
            h.remove()

        n_samples += int(y.shape[0])
        batches += 1
        model.zero_grad(set_to_none=True)

    out: dict[str, dict[str, float]] = {}
    for name in target_layers:
        W = linear_mods[name].weight.detach() if name in linear_mods else None
        B = gevb.B.get(name)
        sub_ang = (
            subspace_principal_angle_deg(W, B)
            if W is not None and B is not None
            else float("nan")
        )
        fa_ang = (
            fa_sums[name] / fa_counts[name]
            if fa_counts[name] > 0
            else float("nan")
        )
        out[name] = {
            "fa_signal_angle_deg": fa_ang,
            "subspace_principal_angle_deg": sub_ang,
            "n_samples": float(fa_counts[name]),
        }
    return out


def find_gevb_signal(signals: Sequence[Any]) -> GEVBVectorEligibility | None:
    """Return the GEVB candidate from a signal list, if present."""
    for s in signals:
        if isinstance(s, GEVBVectorEligibility):
            return s
        if getattr(s, "name", None) == "eligibility_gevb_rf":
            return s  # type: ignore[return-value]
    return None


def alignment_rows_for_csv(
    alignment: Mapping[str, Mapping[str, float]],
    *,
    seed: int,
    step: int,
    snapshot_tag: str,
    epoch: int | None = None,
) -> list[dict[str, Any]]:
    """Flatten alignment dict into CSV-ready rows."""
    rows: list[dict[str, Any]] = []
    for layer, metrics in sorted(alignment.items()):
        rows.append(
            {
                "seed": int(seed),
                "step": int(step),
                "epoch": epoch,
                "snapshot_tag": snapshot_tag,
                "layer": layer,
                "fa_signal_angle_deg": metrics.get("fa_signal_angle_deg"),
                "subspace_principal_angle_deg": metrics.get(
                    "subspace_principal_angle_deg"
                ),
                "n_samples": metrics.get("n_samples"),
            }
        )
    return rows


GEVB_ALIGNMENT_CSV_COLUMNS: tuple[str, ...] = (
    "seed",
    "step",
    "epoch",
    "snapshot_tag",
    "layer",
    "fa_signal_angle_deg",
    "subspace_principal_angle_deg",
    "n_samples",
)
