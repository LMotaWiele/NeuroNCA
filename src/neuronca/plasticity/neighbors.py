"""Same-layer neighbors for local plasticity rules: ``grid`` (default) or ``feature``.

Product default is grid. Feature is a research flag (Exp 1 did not promote it).
See ``docs/design/invariants.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from neuronca.models.dynamic_mlp import HIDDEN_LAYERS, DynamicMLPReasoningNet
from neuronca.plasticity.coords import pairwise_distances


def neighbor_mask_within_radius(
    coords: Tensor,
    active: Tensor,
    radius: float,
) -> Tensor:
    """Bool ``[H, H]``: active pairs within radius; diagonal False."""
    c = coords.float()
    a = active.bool()
    finite = torch.isfinite(c).all(dim=-1) & a
    dist = pairwise_distances(c, c)
    within = (dist <= float(radius)) & finite.unsqueeze(0) & finite.unsqueeze(1)
    within = within.clone()
    within.fill_diagonal_(False)
    return within


def mean_neighbor_count(mask: Tensor, active: Tensor) -> float:
    """Mean number of neighbors per active unit."""
    a = active.bool()
    if not bool(a.any()):
        return 0.0
    counts = mask.float().sum(dim=1)[a]
    return float(counts.mean().item())


def probe_activation_signatures(
    layer_acts: Tensor,
    active: Tensor,
) -> Tensor:
    """Per-unit probe activation signature ``[H, B]`` (zeros for inactive).

    ``layer_acts`` is ``[B, H]`` post-activation on the fixed probe set.
    Signature for unit i is the column of activations over probe samples.
    """
    if layer_acts.ndim != 2:
        raise ValueError(f"layer_acts must be [B,H], got {tuple(layer_acts.shape)}")
    # [H, B]
    sig = layer_acts.detach().float().transpose(0, 1).contiguous()
    a = active.bool().to(device=sig.device)
    sig = sig * a.unsqueeze(1).float()
    return sig


def cosine_similarity_matrix(signatures: Tensor, active: Tensor) -> Tensor:
    """Pairwise cosine similarity ``[H, H]``; inactive rows/cols are -inf."""
    s = signatures.detach().float()
    a = active.bool().to(device=s.device)
    # L2-normalize rows
    norms = torch.linalg.norm(s, dim=1, keepdim=True).clamp(min=1e-8)
    sn = s / norms
    sim = sn @ sn.T  # [H, H]
    # Mask inactive
    active_pair = a.unsqueeze(0) & a.unsqueeze(1)
    sim = torch.where(active_pair, sim, torch.full_like(sim, -1e9))
    sim.fill_diagonal_(-1e9)
    return sim


def neighbor_mask_feature_topk(
    signatures: Tensor,
    active: Tensor,
    k: int,
) -> Tensor:
    """Bool ``[H, H]``: each active unit's top-k cosine-similar active peers."""
    a = active.bool()
    H = int(a.numel())
    k = max(int(k), 0)
    mask = torch.zeros(H, H, dtype=torch.bool, device=a.device)
    if k <= 0 or not bool(a.any()):
        return mask
    sim = cosine_similarity_matrix(signatures, a)
    active_idx = a.nonzero(as_tuple=False).view(-1)
    n_active = int(active_idx.numel())
    # At most n_active - 1 peers
    k_eff = min(k, max(n_active - 1, 0))
    if k_eff <= 0:
        return mask
    # For each active row, top-k among other active (sim already -1e9 on inactive/self)
    # topk over full H is fine given masked sim
    vals, idx = torch.topk(sim, k=k_eff, dim=1, largest=True, sorted=False)
    for i in active_idx.tolist():
        chosen = idx[i]
        # Drop any still-masked entries (if k_eff > available)
        for j, v in zip(chosen.tolist(), vals[i].tolist()):
            if v > -1e8:
                mask[i, j] = True
    mask.fill_diagonal_(False)
    return mask


def neighbor_set_churn(
    prev: Tensor,
    new: Tensor,
    active: Tensor,
) -> float:
    """Mean fraction of previous neighbors lost (per active unit with |N_prev|>0).

    For unit i: |N_prev \\ N_new| / |N_prev|. Units with empty prev neighborhood
    are skipped. Returns 0.0 if no unit has a previous neighbor.
    """
    a = active.bool()
    if prev is None or new is None:
        return 0.0
    prev_b = prev.bool()
    new_b = new.bool()
    fracs: list[float] = []
    for i in a.nonzero(as_tuple=False).view(-1).tolist():
        n_prev = int(prev_b[i].sum().item())
        if n_prev == 0:
            continue
        lost = int((prev_b[i] & ~new_b[i]).sum().item())
        fracs.append(lost / float(n_prev))
    if not fracs:
        return 0.0
    return float(sum(fracs) / len(fracs))


@dataclass
class NeighborRefreshResult:
    """One refresh of the per-layer neighbor graph."""

    metric: str
    masks: dict[str, Tensor] = field(default_factory=dict)
    mean_count_by_layer: dict[str, float] = field(default_factory=dict)
    matched_k_by_layer: dict[str, int] = field(default_factory=dict)
    churn_by_layer: dict[str, float] = field(default_factory=dict)
    mean_churn: float = 0.0
    unstable: bool = False
    grid_mean_count_by_layer: dict[str, float] = field(default_factory=dict)
    similarity: str = "cosine"
    signature_ema_mu: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "mean_count_by_layer": dict(self.mean_count_by_layer),
            "matched_k_by_layer": dict(self.matched_k_by_layer),
            "churn_by_layer": dict(self.churn_by_layer),
            "mean_churn": self.mean_churn,
            "unstable": self.unstable,
            "grid_mean_count_by_layer": dict(self.grid_mean_count_by_layer),
            "similarity": self.similarity,
            "signature_ema_mu": self.signature_ema_mu,
            "notes": list(self.notes),
        }


class NeighborGraph:
    """Stateful neighbor masks refreshed on the plasticity cadence."""

    def __init__(
        self,
        *,
        metric: str = "grid",
        radius: float = 0.35,
        churn_bound: float = 0.5,
        similarity: str = "cosine",
        signature_ema_mu: float = 0.0,
        fixed_k_by_layer: dict[str, int] | None = None,
    ) -> None:
        self.metric = str(metric).strip().lower()
        if self.metric not in ("grid", "feature"):
            raise ValueError(f"neighbor_metric must be 'grid' or 'feature', got {metric!r}")
        self.radius = float(radius)
        self.churn_bound = float(churn_bound)
        self.similarity = str(similarity).strip().lower()
        if self.similarity not in ("cosine", "correlation"):
            raise ValueError(f"unsupported similarity {similarity!r}")
        self.signature_ema_mu = float(signature_ema_mu)
        self.fixed_k_by_layer = dict(fixed_k_by_layer or {})
        self.masks: dict[str, Tensor] = {}
        self._sig_ema: dict[str, Tensor] = {}
        self.last_result: NeighborRefreshResult | None = None
        self.unstable_ever = False
        self.stability_response_applied = False
        self.refresh_count = 0

    def get_mask(self, layer: str) -> Tensor | None:
        return self.masks.get(layer)

    def _signatures_for_layer(
        self, layer: str, layer_acts: Tensor, active: Tensor
    ) -> Tensor:
        raw = probe_activation_signatures(layer_acts, active)
        mu = float(self.signature_ema_mu)
        if mu <= 0.0 or layer not in self._sig_ema:
            self._sig_ema[layer] = raw.clone()
            return raw
        prev = self._sig_ema[layer]
        if prev.shape != raw.shape:
            self._sig_ema[layer] = raw.clone()
            return raw
        ema = mu * prev + (1.0 - mu) * raw
        # Keep inactive rows zeroed
        a = active.bool().to(device=ema.device)
        ema = ema * a.unsqueeze(1).float()
        self._sig_ema[layer] = ema
        return ema

    def refresh(
        self,
        net: DynamicMLPReasoningNet,
        layer_acts: dict[str, Tensor] | None = None,
    ) -> NeighborRefreshResult:
        """Recompute neighbor masks for all hidden layers.

        ``layer_acts`` maps h → ``[B, H]`` probe post-activations. Required
        for ``feature`` metric; ignored for ``grid`` (coords only).
        """
        result = NeighborRefreshResult(
            metric=self.metric,
            similarity=self.similarity,
            signature_ema_mu=self.signature_ema_mu,
        )
        churns: list[float] = []
        new_masks: dict[str, Tensor] = {}

        for h in HIDDEN_LAYERS:
            active = net.active_mask(h).cpu()
            coords = net.coords(h).cpu().float()
            grid_mask = neighbor_mask_within_radius(coords, active, self.radius)
            grid_mean = mean_neighbor_count(grid_mask, active)
            result.grid_mean_count_by_layer[h] = grid_mean

            if self.metric == "grid":
                mask = grid_mask
                k_used = int(round(grid_mean))
            else:
                if layer_acts is None or h not in layer_acts:
                    raise ValueError(
                        "feature neighbor metric requires layer_acts probe tensors"
                    )
                # Matched k = mean grid-neighbor count on this population
                if h in self.fixed_k_by_layer:
                    k = int(self.fixed_k_by_layer[h])
                else:
                    k = max(int(round(grid_mean)), 1)
                k_used = k
                sig = self._signatures_for_layer(h, layer_acts[h].cpu(), active)
                if self.similarity == "correlation":
                    # Center each signature then cosine (= Pearson correlation)
                    a = active.bool()
                    sig_c = sig.clone()
                    for i in a.nonzero(as_tuple=False).view(-1).tolist():
                        row = sig_c[i]
                        sig_c[i] = row - row.mean()
                    sig = sig_c
                mask = neighbor_mask_feature_topk(sig, active, k)

            mean_c = mean_neighbor_count(mask, active)
            result.mean_count_by_layer[h] = mean_c
            result.matched_k_by_layer[h] = k_used

            prev = self.masks.get(h)
            ch = neighbor_set_churn(prev, mask, active) if prev is not None else 0.0
            result.churn_by_layer[h] = ch
            if prev is not None:
                churns.append(ch)
            new_masks[h] = mask

        self.masks = new_masks
        self.refresh_count += 1
        mean_churn = float(sum(churns) / len(churns)) if churns else 0.0
        result.mean_churn = mean_churn
        # First refresh has no prev → churn 0; flag only from refresh 2+
        result.unstable = bool(churns) and mean_churn > self.churn_bound
        if result.unstable:
            self.unstable_ever = True
            result.notes.append(
                f"NEIGHBORHOOD_UNSTABLE: mean_churn={mean_churn:.3f} > "
                f"bound={self.churn_bound}"
            )
        self.last_result = result
        return result

    def apply_stability_response(self) -> dict[str, Any]:
        """Pre-committed response to NEIGHBORHOOD_UNSTABLE.

        Enable EMA-smoothed signatures (mu=0.5) if not already. Caller may
        also slow refresh externally. Idempotent.
        """
        applied: dict[str, Any] = {"already": self.stability_response_applied}
        if self.stability_response_applied:
            return applied
        if self.signature_ema_mu < 0.5:
            self.signature_ema_mu = 0.5
            applied["signature_ema_mu"] = 0.5
        self.stability_response_applied = True
        applied["applied"] = True
        return applied


__all__ = [
    "NeighborGraph",
    "NeighborRefreshResult",
    "neighbor_mask_within_radius",
    "neighbor_mask_feature_topk",
    "probe_activation_signatures",
    "cosine_similarity_matrix",
    "mean_neighbor_count",
    "neighbor_set_churn",
]
