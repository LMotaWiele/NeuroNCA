"""Local structural rules: incentive, eligibility, competitive prune scores.

Pressure/eligibility checks apply **only to hidden endpoints**. Fixed input and
output units are always treated as eligible (plan §Resolved #3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping  # noqa: F401 — used by callers / type docs

import torch
from torch import Tensor

from neuronca.models.dynamic_mlp import (
    HIDDEN_LAYERS,
    LINEAR_LAYERS,
    DynamicMLPReasoningNet,
    _LINEAR_ENDPOINTS,
)
from neuronca.plasticity.coords import linear_decay, pairwise_distances
from neuronca.plasticity.neighbors import neighbor_mask_within_radius
from neuronca.plasticity.threshold import ThresholdPolicy, normalize_target_rates

# Probe-locked defaults (1-epoch ActivationMagnitude mean EMA per layer).
DEFAULT_TARGET_RATES: dict[str, float] = {
    "h1": 0.77,
    "h2": 1.02,
    "h3": 1.28,
}


@dataclass
class PlasticityConfig:
    """Plasticity hyperparameters. Product default ``neighbor_metric`` is grid."""

    radius: float = 0.35
    # Same-layer neighbors: grid (product) | feature (research flag).
    neighbor_metric: str = "grid"  # grid | feature
    neighbor_similarity: str = "cosine"  # cosine | correlation
    neighbor_top_k: int | None = None  # None → match mean grid-neighbor count
    neighbor_churn_bound: float = 0.5
    neighbor_signature_ema_mu: float = 0.0  # 0 = instantaneous signatures
    target_rates: dict[str, float] = field(default_factory=dict)
    target_rate: float | None = None  # test-only broadcast fallback
    tau: float = 0.05
    threshold_kind: str = "fixed"  # fixed | sliding
    mu: float = 0.999
    prune_edge_fraction: float = 0.05
    prune_neuron_z: float = -1.0
    max_new_edges_per_layer: int = 64
    max_new_neurons_per_layer: int = 4
    seed_edges_per_new_neuron: int = 4
    pressure_margin: float = 0.0
    # neither_contraction: ≥1 growth, 0 contraction; not_both_contraction is relaxed.
    edge_eligibility_mode: str = "neither_contraction"
    # Must be >= birth_bias_steps so prune never judges while scaffold is nonzero.
    newborn_grace_steps: int = 3
    newborn_ema_init: str = "layer_target"  # layer_target | zero
    # Temporary pre-act scaffold (not a Parameter); linear decay over bias_steps.
    newborn_birth_bias: float = 0.5
    newborn_birth_bias_steps: int = 3
    # kaiming | parent_perturbed (clone trigger) | gap_imprinted (product birth)
    newborn_weight_init: str = "kaiming"
    newborn_parent_noise: float = 0.1
    newborn_copy_outgoing: bool = True
    newborn_parent_coord_jitter: float = 0.05
    newborn_lr_mult: float = 1.0  # 1.0 = off; scales grads while age < grace
    # activation = act > r*+τ; gated_novelty = novelty ∧ local CE ∧ |N|≥k_min
    neuron_growth_trigger: str = "activation"  # activation | gated_novelty
    novelty_k_min: int = 11
    novelty_rank_fraction: float = 1.0
    # Product ships r_star only; prune_cutoff / straddle stay research flags.
    self_calibrate: dict[str, bool] = field(
        default_factory=lambda: {
            "r_star": False,
            "prune_cutoff": False,  # research only — do not enable in product
            "straddle": False,  # research only — do not enable in product
        }
    )
    self_cal_mu_r_star: float = 0.999
    self_cal_mu_prune: float = 0.999
    self_cal_mu_straddle: float = 0.99
    # None = update every plasticity step; int N = hybrid every N steps.
    self_cal_hybrid_period: int | None = None
    # Target left-tail quantile for self-cal prune cut (≈ Φ(-1)).
    self_cal_prune_quantile: float = 0.1587

    def __post_init__(self) -> None:
        # Invariant: prune must not judge while birth-bias scaffold is nonzero.
        if int(self.newborn_grace_steps) < int(self.newborn_birth_bias_steps):
            raise ValueError(
                "newborn_grace_steps must be >= newborn_birth_bias_steps so "
                "judgment never mixes real weight-driven signal with residual "
                f"scaffold bias (got grace={self.newborn_grace_steps}, "
                f"bias_steps={self.newborn_birth_bias_steps})"
            )

    def resolved_target_rates(self) -> dict[str, float]:
        if self.target_rates:
            return normalize_target_rates(target_rates=self.target_rates)
        if self.target_rate is not None:
            return normalize_target_rates(self.target_rate)
        return dict(DEFAULT_TARGET_RATES)

    def self_calibrate_flags(self) -> dict[str, bool]:
        raw = dict(self.self_calibrate or {})
        return {
            "r_star": bool(raw.get("r_star", False)),
            "prune_cutoff": bool(raw.get("prune_cutoff", False)),
            "straddle": bool(raw.get("straddle", False)),
        }


def population_signals_from_activation(
    net: DynamicMLPReasoningNet,
    neuron_scores: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Map ActivationMagnitude layer keys to population signal vectors.

    ``neuron_scores`` uses Linear names (fc*) → postsynaptic ``[n_out]``.
    Hidden populations use fc1/fc2/fc3 post scores. Input uses mean abs from
    ``act_in`` EMA if present under a synthetic key; otherwise zeros with
    live batch fallback left to caller.

    For edge incentive we need pre and post signals. Pre for fc1 is input:
    we store under ``in`` if ``neuron_scores`` has ``act_in`` or ``in``.
    """
    out: dict[str, Tensor] = {}
    # Hidden = postsynaptic of each fc*
    out["h1"] = neuron_scores["fc1"].detach().float().cpu()
    out["h2"] = neuron_scores["fc2"].detach().float().cpu()
    out["h3"] = neuron_scores["fc3"].detach().float().cpu()
    # Output posts of fc_out
    out["out"] = neuron_scores["fc_out"].detach().float().cpu()
    if "in" in neuron_scores:
        out["in"] = neuron_scores["in"].detach().float().cpu()
    elif "act_in" in neuron_scores:
        out["in"] = neuron_scores["act_in"].detach().float().cpu()
    else:
        # Placeholder; orchestrator should inject input EMA.
        out["in"] = torch.zeros(net.input_dim, dtype=torch.float32)
    return out


def local_zscore(values: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    """Z-score of ``values`` among positions where ``mask`` is True; 0 elsewhere."""
    out = torch.zeros_like(values, dtype=torch.float32)
    if not bool(mask.any()):
        return out
    v = values[mask].float()
    mean = v.mean()
    std = v.std(unbiased=False)
    if float(std) < eps:
        out[mask] = 0.0
    else:
        out[mask] = (values[mask].float() - mean) / (std + eps)
    return out


def edge_incentive_matrix(
    net: DynamicMLPReasoningNet,
    layer: str,
    signals: dict[str, Tensor],
    radius: float,
) -> Tensor:
    """Incentive for all out×in pairs of ``layer``; 0 if d > R or inactive.

    ``incentive(i,j) = s_post[i] * s_pre[j] * decay(d)``.
    """
    pre_pop, post_pop = _LINEAR_ENDPOINTS[layer]
    pre_a = net.population_active(pre_pop).cpu()
    post_a = net.population_active(post_pop).cpu()
    pre_c = net.coords(pre_pop).cpu().float()
    post_c = net.coords(post_pop).cpu().float()
    # Fixed populations always finite; hidden may be NaN when inactive.
    s_pre = signals[pre_pop]
    s_post = signals[post_pop]
    dist = pairwise_distances(post_c, pre_c)  # [n_out, n_in]
    # Inactive / NaN coords → large distance
    nan_post = torch.isnan(post_c).any(dim=-1)
    nan_pre = torch.isnan(pre_c).any(dim=-1)
    dist = dist.clone()
    dist[nan_post] = float("inf")
    dist[:, nan_pre] = float("inf")
    decay = linear_decay(dist.clamp(max=radius * 10), radius)
    # linear_decay clamps; force zero beyond R
    decay = torch.where(dist <= radius, decay, torch.zeros_like(decay))
    incent = s_post.unsqueeze(1) * s_pre.unsqueeze(0) * decay
    incent = incent * post_a.unsqueeze(1).float() * pre_a.unsqueeze(0).float()
    incent = torch.nan_to_num(incent, nan=0.0, posinf=0.0, neginf=0.0)
    return incent


def hidden_endpoints_for_edge(layer: str, out_idx: int, in_idx: int) -> list[tuple[str, int]]:
    """Return hidden (population, index) endpoints of an edge (0, 1, or 2)."""
    pre, post = _LINEAR_ENDPOINTS[layer]
    ends: list[tuple[str, int]] = []
    if pre in HIDDEN_LAYERS:
        ends.append((pre, int(in_idx)))
    if post in HIDDEN_LAYERS:
        ends.append((post, int(out_idx)))
    return ends


def edge_eligibility(
    layer: str,
    out_idx: int,
    in_idx: int,
    policy: ThresholdPolicy,
    *,
    mode: str = "neither_contraction",
) -> tuple[bool, list[tuple[str, int]]]:
    """Pressure gate: only hidden endpoints; fixed I/O always eligible.

    Modes
    -----
    ``neither_contraction`` (strict / plan default)
        ≥1 hidden endpoint under growth pressure; **no** hidden endpoint under
        contraction pressure.
    ``not_both_contraction`` (relaxed diagnostic)
        ≥1 hidden endpoint under growth pressure; reject only if **every**
        hidden endpoint is under contraction (a single contracting partner
        no longer vetoes a growing one).

    Returns:
        (eligible, list of hidden endpoints currently under growth pressure).
    """
    ends = hidden_endpoints_for_edge(layer, out_idx, in_idx)
    if not ends:
        return False, []
    key = mode.strip().lower()
    growth_ends: list[tuple[str, int]] = []
    n_contract = 0
    for pop, idx in ends:
        contracting = bool(policy.contraction_pressure(pop)[idx])
        growing = bool(policy.growth_pressure(pop)[idx])
        if contracting:
            n_contract += 1
        if growing:
            growth_ends.append((pop, idx))
    if not growth_ends:
        return False, []
    if key in ("neither_contraction", "strict", "default"):
        if n_contract > 0:
            return False, []
        return True, growth_ends
    if key in ("not_both_contraction", "relaxed", "not_both"):
        # Veto only when all hidden endpoints are contracting.
        if n_contract >= len(ends) and len(ends) > 0:
            # If every hidden end contracts, block — even if one also flagged
            # growth (mutually exclusive in practice for FixedThreshold).
            return False, []
        # With ≥1 growth already required: allow if not every end contracts.
        # Equivalent: block only when n_contract == len(ends).
        if n_contract == len(ends):
            return False, []
        return True, growth_ends
    raise ValueError(
        f"unknown edge_eligibility mode {mode!r}; expected "
        f"'neither_contraction' or 'not_both_contraction'"
    )


def competitive_edge_zscores(
    net: DynamicMLPReasoningNet,
    layer: str,
    signals: dict[str, Tensor],
    radius: float,
) -> Tensor:
    """Locally normalized contribution scores for existing edges ``[out, in]``.

    Raw score = s_post * s_pre; z-scored among edges whose post unit is within
    radius of each other is approximated by global active-edge z-score within
    the layer for Phase 1a simplicity (neighborhood z can be refined later).
    Plan allows local neighborhood within R; we z-score among edges incident to
    posts within R of a reference post — vectorized approximation: z-score all
    active edges in the layer (still relative, not raw magnitude alone).
    """
    pre_pop, post_pop = _LINEAR_ENDPOINTS[layer]
    em = net.edge_mask(layer).cpu()
    s_pre = signals[pre_pop]
    s_post = signals[post_pop]
    raw = s_post.unsqueeze(1) * s_pre.unsqueeze(0)
    raw = raw * em.float()
    return local_zscore(raw.reshape(-1), em.reshape(-1)).reshape_as(raw)


def competitive_neuron_zscores(
    net: DynamicMLPReasoningNet,
    layer: str,
    signals: dict[str, Tensor],
    neighbor_mask: Tensor | None = None,
) -> Tensor:
    """Z-score of neuron signals among neighbors (or whole layer if mask is None).

    When ``neighbor_mask`` is provided (Experiment 1 / design-intent local
    competition), each unit's z is computed among ``N(i) ∪ {i}`` active peers
    under the current neighbor definition. Without a mask, falls back to the
    historical whole-layer z-score among active units.
    """
    s = signals[layer].float().cpu()
    a = net.active_mask(layer).cpu().bool()
    if neighbor_mask is None:
        return local_zscore(s, a)
    nbr = neighbor_mask.bool().cpu()
    if nbr.shape[0] != s.numel() or nbr.shape[1] != s.numel():
        raise ValueError(
            f"neighbor_mask shape {tuple(nbr.shape)} != [{s.numel()}, {s.numel()}]"
        )
    z = torch.zeros_like(s)
    active_idx = a.nonzero(as_tuple=False).view(-1).tolist()
    for i in active_idx:
        members = nbr[i].clone()
        members[i] = True
        members = members & a
        mcount = int(members.sum().item())
        if mcount < 2:
            # Isolated / under-floor: no local competition signal → z = 0
            z[i] = 0.0
            continue
        vals = s[members]
        mean = vals.mean()
        std = vals.std(unbiased=False)
        if float(std) < 1e-8:
            z[i] = 0.0
        else:
            z[i] = (s[i] - mean) / (std + 1e-8)
    return z


def select_edges_to_prune(
    net: DynamicMLPReasoningNet,
    layer: str,
    zscores: Tensor,
    fraction: float,
) -> list[tuple[int, int]]:
    """Bottom ``fraction`` of existing edges by z-score."""
    em = net.edge_mask(layer).cpu()
    if not bool(em.any()) or fraction <= 0:
        return []
    idx = em.nonzero(as_tuple=False)  # [E, 2]
    zs = zscores[em]
    n_drop = max(1, int(float(zs.numel()) * fraction)) if zs.numel() else 0
    if n_drop <= 0:
        return []
    n_drop = min(n_drop, int(zs.numel()))
    order = torch.argsort(zs)[:n_drop]
    pairs = idx[order]
    return [(int(pairs[i, 0]), int(pairs[i, 1])) for i in range(pairs.shape[0])]


def select_neurons_to_prune(
    net: DynamicMLPReasoningNet,
    layer: str,
    zscores: Tensor,
    policy: ThresholdPolicy,
    z_cut: float,
    *,
    grace_steps: int = 0,
) -> list[int]:
    """Active neurons with contraction pressure and z < z_cut.

    ``grace_steps``: exclude slots younger than this many plasticity steps
    (newborn grace; does not apply to init-origin age once steps advance).
    """
    a = net.active_mask(layer).cpu()
    contr = policy.contraction_pressure(layer).cpu()
    cand = a & contr & (zscores < z_cut)
    if grace_steps > 0:
        birth = net.birth_step(layer).cpu()
        step = int(net.cooldown.plasticity_step)
        # age = step - birth; eligible when age >= grace_steps
        age_ok = (birth >= 0) & ((step - birth) >= int(grace_steps))
        cand = cand & age_ok
    return cand.nonzero(as_tuple=False).view(-1).tolist()


@dataclass
class EdgeGrowStats:
    """Diagnostics for standalone incentive edge formation (pre/post τ).

    Pair IDs are ``(layer, out_idx, in_idx)`` and are collected only when
    ``collect_pairs=True`` on :func:`select_edges_to_grow` (same enumeration
    loop the orchestrator uses to grow edges — not a parallel taxonomy).
    """

    # Absent edges among active endpoints with incentive > 0 (implies d ≤ R
    # under linear decay; zero-signal pairs excluded).
    missing_in_radius: int = 0
    # Blocked solely by cooldown (still missing + incentive>0).
    blocked_cooldown: int = 0
    # Fail homeostatic eligibility after clearing cooldown.
    blocked_eligibility: int = 0
    # Of those: pass cooldown + homeostatic eligibility gate — evaluated vs τ.
    candidates_evaluated: int = 0
    # Of evaluated: incentive clears threshold (pre rate-cap).
    threshold_passes: int = 0
    # Strict growth+no_contraction pattern (independent of eligibility mode),
    # counted for every missing_in_radius pair in this same loop.
    growth_no_contraction: int = 0
    # Subset: growth_no_contraction and currently on cooldown.
    growth_no_contraction_on_cooldown: int = 0
    # Subset: growth_no_contraction and not on cooldown (should enter eligibility).
    growth_no_contraction_off_cooldown: int = 0
    # Optional pair ID lists (only if collect_pairs=True).
    pairs_growth_no_contraction: list[tuple[str, int, int]] = field(
        default_factory=list
    )
    pairs_growth_no_contraction_on_cooldown: list[tuple[str, int, int]] = field(
        default_factory=list
    )
    pairs_candidates_evaluated: list[tuple[str, int, int]] = field(
        default_factory=list
    )


def _strict_growth_no_contraction(
    layer: str,
    out_idx: int,
    in_idx: int,
    policy: ThresholdPolicy,
) -> bool:
    """True iff hidden ends have ≥1 growth and 0 contraction (strict pattern)."""
    ends = hidden_endpoints_for_edge(layer, out_idx, in_idx)
    if not ends:
        return False
    n_g = 0
    n_c = 0
    for pop, idx in ends:
        if bool(policy.growth_pressure(pop)[idx]):
            n_g += 1
        if bool(policy.contraction_pressure(pop)[idx]):
            n_c += 1
    return n_g >= 1 and n_c == 0


def select_edges_to_grow(
    net: DynamicMLPReasoningNet,
    layer: str,
    incentives: Tensor,
    policy: ThresholdPolicy,
    max_new: int,
    *,
    eligibility_mode: str = "neither_contraction",
    return_stats: bool = False,
    return_precap_count: bool = False,
    collect_pairs: bool = False,
) -> (
    list[tuple[int, int, float]]
    | tuple[list[tuple[int, int, float]], int]
    | tuple[list[tuple[int, int, float]], EdgeGrowStats]
):
    """Rank absent eligible edges by incentive; return up to max_new triples.

    This is the **only** enumeration used by the orchestrator for standalone
    incentive edge growth. Diagnostics must call this function (or share its
    loop), not a parallel pair walk.

    ``return_stats``: also return :class:`EdgeGrowStats`.
    ``return_precap_count``: legacy — return threshold_passes count only.
    ``collect_pairs``: populate pair-ID lists on the stats object (heavier).
    """
    em = net.edge_mask(layer).cpu()
    pre_pop, post_pop = _LINEAR_ENDPOINTS[layer]
    pre_a = net.population_active(pre_pop).cpu()
    post_a = net.population_active(post_pop).cpu()
    candidates: list[tuple[int, int, float]] = []
    stats = EdgeGrowStats()
    # Bound work: missing edges with incentive > 0 ⇒ in radius under linear decay.
    pos = (incentives > 0) & (~em) & post_a.unsqueeze(1) & pre_a.unsqueeze(0)
    coords = pos.nonzero(as_tuple=False)
    stats.missing_in_radius = int(coords.shape[0])
    for k in range(coords.shape[0]):
        oi = int(coords[k, 0])
        ii = int(coords[k, 1])
        pair_id = (layer, oi, ii)
        gnc = _strict_growth_no_contraction(layer, oi, ii, policy)
        if gnc:
            stats.growth_no_contraction += 1
            if collect_pairs:
                stats.pairs_growth_no_contraction.append(pair_id)

        # Cooldown check without consuming/mutating via a peek when possible.
        # edge_eligible mutates expired entries and counts blocks — same as live path.
        on_cd = not net.cooldown.edge_eligible(layer, oi, ii)
        if on_cd:
            stats.blocked_cooldown += 1
            if gnc:
                stats.growth_no_contraction_on_cooldown += 1
                if collect_pairs:
                    stats.pairs_growth_no_contraction_on_cooldown.append(pair_id)
            continue
        if gnc:
            stats.growth_no_contraction_off_cooldown += 1

        ok, growth_ends = edge_eligibility(
            layer, oi, ii, policy, mode=eligibility_mode
        )
        if not ok:
            stats.blocked_eligibility += 1
            continue
        stats.candidates_evaluated += 1
        if collect_pairs:
            stats.pairs_candidates_evaluated.append(pair_id)
        thr = policy.edge_threshold_for_pair(growth_ends)
        inc = float(incentives[oi, ii].item())
        if inc > thr:
            candidates.append((oi, ii, inc))
    candidates.sort(key=lambda t: t[2], reverse=True)
    stats.threshold_passes = len(candidates)
    capped = candidates[: max(0, max_new)]
    if return_stats:
        return capped, stats
    if return_precap_count:
        return capped, stats.threshold_passes
    return capped


def select_neurons_to_grow(
    net: DynamicMLPReasoningNet,
    layer: str,
    policy: ThresholdPolicy,
    max_new: int,
    *,
    return_precap_count: bool = False,
) -> list[int] | tuple[list[int], int]:
    """Triggers under growth pressure; return up to max_new (post free-slot/rate cap).

    When ``return_precap_count`` is True, also return the raw trigger count
    (active + growth pressure) before rate/free-slot caps.

    Legacy path: ``act > r*_ℓ + τ`` via :meth:`ThresholdPolicy.growth_pressure`.
    For gated novelty, use :func:`select_neurons_to_grow_gated_novelty`.
    """
    g = policy.growth_pressure(layer).cpu()
    a = net.active_mask(layer).cpu()
    triggers = (g & a).nonzero(as_tuple=False).view(-1).tolist()
    precap = len(triggers)
    if net.at_capacity(layer) or net.free_slot_count(layer) == 0:
        capped: list[int] = []
    else:
        n = min(len(triggers), max_new, net.free_slot_count(layer))
        capped = triggers[:n]
    if return_precap_count:
        return capped, precap
    return capped


@dataclass
class GatedNoveltyLayerStats:
    """Per-layer gated-novelty diagnostics (pre-registered three-part rule)."""

    layer: str
    n_active: int = 0
    n_under_floor: int = 0  # |N| < k_min — excluded, logged, monitored
    n_density_pass: int = 0
    n_ce_gate_pass: int = 0
    n_candidates: int = 0  # density ∧ CE ∧ ranked novelty pool
    under_floor_slots: list[int] = field(default_factory=list)
    # Ranked trigger slots (highest novelty first) before rate/free caps.
    ranked_triggers: list[int] = field(default_factory=list)
    # Full ranked candidates: (slot, mean_nbr_max, local_ce_excess, n_neighbors).
    candidates: list[tuple[int, float, float, int]] = field(default_factory=list)
    # Compact metrics for logs (top 32).
    candidate_metrics: list[dict[str, float]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "layer": self.layer,
            "n_active": self.n_active,
            "n_under_floor": self.n_under_floor,
            "n_density_pass": self.n_density_pass,
            "n_ce_gate_pass": self.n_ce_gate_pass,
            "n_candidates": self.n_candidates,
            "under_floor_slots": list(self.under_floor_slots),
            "ranked_triggers": list(self.ranked_triggers),
            "candidate_metrics": list(self.candidate_metrics),
        }


@dataclass
class GatedNoveltyContext:
    """Probe-time gated-novelty evaluation for one plasticity step.

    Optional probe tensors support **gap_imprinted** birth (not parent clone):
    pre-activations per hidden layer and per-sample CE, used to form the
    residual direction on CE-excess / low-neighbor-max samples.
    """

    by_layer: dict[str, GatedNoveltyLayerStats] = field(default_factory=dict)
    # Probe tensors for gap imprinting (CPU).
    ce: Tensor | None = None  # [B]
    # post-activations h → [B, H] (same as scoring)
    post_acts: dict[str, Tensor] = field(default_factory=dict)
    # pre-activations for each hidden layer's *incoming* linear: h → [B, fan_in]
    pre_acts: dict[str, Tensor] = field(default_factory=dict)
    # Per (layer, slot) gap sample mask [B] for candidates (bool).
    gap_masks: dict[tuple[str, int], Tensor] = field(default_factory=dict)

    def stats_for(self, layer: str) -> GatedNoveltyLayerStats:
        if layer not in self.by_layer:
            self.by_layer[layer] = GatedNoveltyLayerStats(layer=layer)
        return self.by_layer[layer]

    def as_dict(self) -> dict:
        return {h: s.as_dict() for h, s in self.by_layer.items()}


# neighbor_mask_within_radius re-exported from neighbors for backward compatibility


def gap_imprint_direction(
    pre_acts: Tensor,
    gap_mask: Tensor,
    ce: Tensor | None = None,
    *,
    subtract_global_mean: bool = True,
) -> Tensor:
    """Content direction of the coverage gap from probe pre-activations.

    On gap samples (low neighbor coverage / CE-excess band): CE-weighted mean
    pre-activation, minus global mean pre (residual = what is special about the
    gap). This is **not** a copy of any existing unit's weight row.

    Returns:
        Float vector ``[fan_in]`` (unnormalized residual direction).
    """
    pre = pre_acts.detach().float()
    if pre.ndim != 2:
        raise ValueError(f"pre_acts must be [B, fan_in], got {tuple(pre.shape)}")
    mask = gap_mask.bool().to(device=pre.device)
    if not bool(mask.any()):
        raise ValueError("gap_mask has no True samples")
    g = pre[mask]  # [Bg, fan_in]
    if ce is not None:
        w = ce.detach().float().to(device=pre.device)[mask].clamp(min=0.0)
        w = w / w.sum().clamp(min=1e-8)
        gap_mean = (g * w.unsqueeze(1)).sum(dim=0)
    else:
        gap_mean = g.mean(dim=0)
    if subtract_global_mean:
        return gap_mean - pre.mean(dim=0)
    return gap_mean


def per_dimension_separation_scores(
    pre_acts: Tensor,
    gap_mask: Tensor,
    *,
    residual: Tensor | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """Per-input-dim gap vs non-gap separating power (not magnitude).

    For each dimension j:
        delta_j = mean(pre_j | gap) - mean(pre_j | non-gap)
        score_j = |delta_j| / (std(pre_j) + eps)

    If ``residual`` is provided, dimensions whose delta opposes the residual
    sign are zeroed (only keep directions that help the imprint functional).

    This is the correct edge-selection objective for gap imprint: high |W|
    can mark high-dynamic-range dimensions that are *not* diagnostic of the
    gap. Separating power targets Cohen-style per-dim contribution.
    """
    pre = pre_acts.detach().float()
    m = gap_mask.bool().to(device=pre.device)
    if pre.ndim != 2:
        raise ValueError(f"pre_acts must be [B, fan_in], got {tuple(pre.shape)}")
    if not bool(m.any()) or not bool((~m).any()):
        return torch.zeros(pre.shape[1], dtype=torch.float32)
    gap = pre[m]
    nongap = pre[~m]
    delta = gap.mean(dim=0) - nongap.mean(dim=0)
    std = pre.std(dim=0, unbiased=False).clamp(min=eps)
    score = delta.abs() / std
    if residual is not None:
        r = residual.detach().float().to(device=score.device).reshape(-1)
        if r.numel() != score.numel():
            raise ValueError("residual length must match fan_in")
        # Keep dims that pull the same way as the residual mean-diff.
        aligned = (delta * r) > 0
        score = torch.where(aligned, score, torch.zeros_like(score))
    return score


def scale_direction_to_kaiming_relu(
    direction: Tensor,
    *,
    fan_in: int | None = None,
) -> Tensor:
    """Rescale a direction vector to Kaiming-ReLU RMS.

    ``fan_in`` defaults to ``len(direction)``. For sparse open-edge realization
    pass ``fan_in = n_open`` so RMS matches the *effective* fan, not the dense
    ambient dimension that is mostly masked.
    """
    d = direction.detach().float().reshape(-1)
    n = max(int(fan_in) if fan_in is not None else int(d.numel()), 1)
    target_std = (2.0 / float(n)) ** 0.5
    s = float(d.std(unbiased=False).clamp(min=1e-8).item())
    return d * (target_std / s)


def gated_novelty_scores_for_layer(
    acts: Tensor,
    ce: Tensor,
    coords: Tensor,
    active: Tensor,
    *,
    radius: float,
    k_min: int,
    neighbor_mask: Tensor | None = None,
) -> tuple[GatedNoveltyLayerStats, list[tuple[int, float, float, int]], dict[int, Tensor]]:
    """Score active units for gated novelty on one layer.

    Args:
        acts: ``[B, H]`` post-activation (absolute magnitude used).
        ce: ``[B]`` per-sample cross-entropy.
        coords: ``[H, 2]`` unit-square coords (used only when neighbor_mask is None).
        active: ``[H]`` bool.
        radius: sensing radius R (grid fallback).
        k_min: density floor (exclude if |N| < k_min).
        neighbor_mask: optional precomputed ``[H, H]`` neighbor graph. When set,
            this is the sole neighbor definition (grid or feature). When None,
            falls back to grid radius on ``coords``.

    Returns:
        stats, candidates (slot, mean_nbr_max, local_ce_excess, n_neighbors)
        sorted by novelty, and ``gap_masks`` mapping slot → bool ``[B]`` gap mask.
    """
    if acts.ndim != 2:
        raise ValueError(f"acts must be [B,H], got {tuple(acts.shape)}")
    if ce.ndim != 1 or ce.shape[0] != acts.shape[0]:
        raise ValueError("ce must be [B] matching acts batch")
    H = acts.shape[1]
    a = active.bool().cpu()
    if a.numel() != H:
        raise ValueError("active length must match H")
    stats = GatedNoveltyLayerStats(layer="")  # caller sets layer
    stats.n_active = int(a.sum().item())
    if stats.n_active == 0:
        return stats, [], {}

    if neighbor_mask is not None:
        nbr = neighbor_mask.bool().cpu()
        if nbr.shape != (H, H):
            raise ValueError(f"neighbor_mask must be [{H},{H}], got {tuple(nbr.shape)}")
        # Ensure inactive pairs are excluded
        nbr = nbr & a.unsqueeze(0) & a.unsqueeze(1)
        nbr = nbr.clone()
        nbr.fill_diagonal_(False)
    else:
        nbr = neighbor_mask_within_radius(coords.cpu().float(), a, radius)
    abs_act = acts.detach().float().abs().cpu()  # [B, H]
    ce_cpu = ce.detach().float().cpu()
    mean_ce = float(ce_cpu.mean().item()) if ce_cpu.numel() else 0.0
    B = abs_act.shape[0]

    candidates: list[tuple[int, float, float, int]] = []
    # slot → gap sample mask (for gap_imprinted birth)
    gap_masks: dict[int, Tensor] = {}
    active_idx = a.nonzero(as_tuple=False).view(-1).tolist()
    for slot in active_idx:
        n_mask = nbr[slot]
        n_count = int(n_mask.sum().item())
        if n_count < int(k_min):
            stats.n_under_floor += 1
            stats.under_floor_slots.append(int(slot))
            continue
        stats.n_density_pass += 1
        # Per-sample max |act| over live neighbors.
        nbr_idx = n_mask.nonzero(as_tuple=False).view(-1)
        nbr_acts = abs_act[:, nbr_idx]  # [B, n_count]
        nbr_max = nbr_acts.max(dim=1).values  # [B]
        mean_nbr_max = float(nbr_max.mean().item())
        # Local CE excess on gap: bottom-quartile nbr_max samples.
        if B < 4:
            q = float(nbr_max.median().item())
            gap_mask = nbr_max <= q
        else:
            q = float(torch.quantile(nbr_max, 0.25).item())
            gap_mask = nbr_max <= q
        if not bool(gap_mask.any()):
            continue
        ce_gap = float(ce_cpu[gap_mask].mean().item())
        local_ce_excess = ce_gap - mean_ce
        if local_ce_excess <= 0.0:
            continue
        stats.n_ce_gate_pass += 1
        candidates.append((int(slot), mean_nbr_max, local_ce_excess, n_count))
        gap_masks[int(slot)] = gap_mask.clone()

    # Highest novelty first = lowest mean neighbor-max.
    candidates.sort(key=lambda t: t[1])
    stats.n_candidates = len(candidates)
    stats.candidates = list(candidates)
    stats.ranked_triggers = [c[0] for c in candidates]
    stats.candidate_metrics = [
        {
            "slot": float(s),
            "mean_nbr_max": float(mnm),
            "local_ce_excess": float(ce_ex),
            "n_neighbors": float(nn),
            "novelty": float(-mnm),
        }
        for s, mnm, ce_ex, nn in candidates[:32]
    ]
    return stats, candidates, gap_masks


def select_neurons_to_grow_gated_novelty(
    net: DynamicMLPReasoningNet,
    layer: str,
    candidates: list[tuple[int, float, float, int]],
    max_new: int,
    *,
    rank_fraction: float = 1.0,
    return_precap_count: bool = False,
) -> list[int] | tuple[list[int], int]:
    """Rate/free-slot cap on ranked gated-novelty candidates (already density∧CE).

    ``candidates``: sorted highest-novelty first from
    :func:`gated_novelty_scores_for_layer`.
    """
    precap = len(candidates)
    if rank_fraction < 1.0 and precap > 0:
        keep = max(1, int(float(precap) * float(rank_fraction)))
        pool = candidates[:keep]
    else:
        pool = candidates
    triggers = [c[0] for c in pool]
    precap = len(triggers)
    if net.at_capacity(layer) or net.free_slot_count(layer) == 0:
        capped: list[int] = []
    else:
        n = min(len(triggers), max_new, net.free_slot_count(layer))
        capped = triggers[:n]
    if return_precap_count:
        return capped, precap
    return capped


def evaluate_gated_novelty_context(
    net: DynamicMLPReasoningNet,
    layer_acts: dict[str, Tensor],
    ce: Tensor,
    *,
    radius: float,
    k_min: int,
    pre_acts: dict[str, Tensor] | None = None,
    neighbor_masks: dict[str, Tensor] | None = None,
) -> GatedNoveltyContext:
    """Build a full :class:`GatedNoveltyContext` from probe activations + CE.

    ``layer_acts`` maps ``h1|h2|h3`` → ``[B, H_max]`` post-activations.
    ``pre_acts`` maps ``h1|h2|h3`` → ``[B, fan_in]`` pre-activations of the
    incoming linear (for gap_imprinted birth).
    ``neighbor_masks`` optional per-layer ``[H,H]`` graphs (grid or feature).
    """
    ctx = GatedNoveltyContext()
    ctx.ce = ce.detach().float().cpu()
    ctx.post_acts = {h: layer_acts[h].detach().float().cpu() for h in layer_acts}
    if pre_acts is not None:
        ctx.pre_acts = {h: pre_acts[h].detach().float().cpu() for h in pre_acts}
    for h in HIDDEN_LAYERS:
        if h not in layer_acts:
            continue
        acts = layer_acts[h]
        active = net.active_mask(h).cpu()
        coords = net.coords(h).cpu()
        nmask = None
        if neighbor_masks is not None and h in neighbor_masks:
            nmask = neighbor_masks[h]
        stats, _cands, gap_masks = gated_novelty_scores_for_layer(
            acts,
            ce,
            coords,
            active,
            radius=radius,
            k_min=k_min,
            neighbor_mask=nmask,
        )
        stats.layer = h
        ctx.by_layer[h] = stats
        for slot, mask in gap_masks.items():
            ctx.gap_masks[(h, int(slot))] = mask.cpu()
    return ctx


__all__ = [
    "PlasticityConfig",
    "EdgeGrowStats",
    "GatedNoveltyLayerStats",
    "GatedNoveltyContext",
    "population_signals_from_activation",
    "local_zscore",
    "edge_incentive_matrix",
    "edge_eligibility",
    "hidden_endpoints_for_edge",
    "competitive_edge_zscores",
    "competitive_neuron_zscores",
    "select_edges_to_prune",
    "select_neurons_to_prune",
    "select_edges_to_grow",
    "select_neurons_to_grow",
    "select_neurons_to_grow_gated_novelty",
    "neighbor_mask_within_radius",
    "gated_novelty_scores_for_layer",
    "gap_imprint_direction",
    "per_dimension_separation_scores",
    "scale_direction_to_kaiming_relu",
    "evaluate_gated_novelty_context",
    "LINEAR_LAYERS",
]
