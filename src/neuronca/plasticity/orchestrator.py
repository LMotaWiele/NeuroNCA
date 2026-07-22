"""Plasticity orchestrator: prune → cooldown → grow edges → grow neurons.

Structural decisions use only local activation signals and rules — never
gradients or task labels. Cadence is once per call (typically once per epoch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

from neuronca.models.dynamic_mlp import (
    HIDDEN_LAYERS,
    LINEAR_LAYERS,
    DynamicMLPReasoningNet,
    _LINEAR_ENDPOINTS,
)
from neuronca.plasticity.masks import (
    EDGE_ORIGIN_GAP_IMPRINT,
    EDGE_ORIGIN_INCENTIVE,
    EDGE_ORIGIN_INIT,
    NEURON_ORIGIN_GROWN,
)
from neuronca.plasticity.neighbors import NeighborGraph
from neuronca.plasticity.rules import (
    GatedNoveltyContext,
    PlasticityConfig,
    competitive_edge_zscores,
    competitive_neuron_zscores,
    edge_incentive_matrix,
    gap_imprint_direction,
    per_dimension_separation_scores,
    population_signals_from_activation,
    scale_direction_to_kaiming_relu,
    select_edges_to_grow,
    select_edges_to_prune,
    select_neurons_to_grow,
    select_neurons_to_grow_gated_novelty,
    select_neurons_to_prune,
)
from neuronca.plasticity.self_calibrate import SelfCalibrateState
from neuronca.plasticity.threshold import (
    SlidingThreshold,
    ThresholdPolicy,
    build_threshold_policy,
)


@dataclass
class StructureEventLog:
    """Per-step structure diagnostics (including prune-by-origin)."""

    step: int = 0
    epoch: int | None = None
    hidden_counts: dict[str, int] = field(default_factory=dict)
    edge_counts: dict[str, Any] = field(default_factory=dict)
    edges_pruned_init_dense: int = 0
    edges_pruned_incentive_grown: int = 0
    # Split edge removals: competitive z-score path vs neuron-deactivation collateral.
    edges_pruned_competitive: int = 0
    edges_pruned_collateral: int = 0
    edges_grown: int = 0
    # Split growth: incentive+τ path vs birth-seed path (seed may bypass τ).
    edges_grown_incentive: int = 0
    edges_grown_seed: int = 0
    neurons_pruned: int = 0
    neurons_grown: int = 0
    # Pre-rate-cap candidate counts (τ wiring diagnostics).
    edge_grow_precap: int = 0  # threshold passes (legacy name)
    edge_missing_in_radius: int = 0  # gaps with incentive>0 (d≤R)
    edge_candidates_evaluated: int = 0  # eligibility-pass; may fail τ
    edge_blocked_cooldown: int = 0
    edge_blocked_eligibility: int = 0
    edge_growth_no_contraction: int = 0
    edge_gnc_on_cooldown: int = 0
    edge_gnc_off_cooldown: int = 0
    # Marginal homeostatic pressure counts (active neurons), independent of pairing.
    n_growth_pressure: int = 0
    n_contraction_pressure: int = 0
    n_active_hidden: int = 0
    growth_pressure_by_layer: dict[str, int] = field(default_factory=dict)
    contraction_pressure_by_layer: dict[str, int] = field(default_factory=dict)
    neuron_grow_precap: int = 0
    edge_grow_postcap_attempts: int = 0
    neuron_grow_postcap_attempts: int = 0
    cap_hits: dict[str, bool] = field(default_factory=dict)
    cooldown_blocks: int = 0
    same_step_free_reuse_blocked: int = 0
    # Pair IDs from select_edges_to_grow (same loop); only if collect_pairs.
    pairs_growth_no_contraction: list[tuple[str, int, int]] = field(default_factory=list)
    pairs_gnc_on_cooldown: list[tuple[str, int, int]] = field(default_factory=list)
    pairs_candidates_evaluated: list[tuple[str, int, int]] = field(default_factory=list)
    # Structure event keys for thrash analysis (pre-registered stability bar).
    neuron_prune_keys: list[tuple[str, int]] = field(default_factory=list)
    # Rich prune records for structure visualization (coords captured at death).
    neuron_prune_records: list[dict[str, Any]] = field(default_factory=list)
    neuron_grow_keys: list[tuple[str, int]] = field(default_factory=list)
    edge_prune_keys: list[tuple[str, int, int]] = field(default_factory=list)
    edge_grow_keys: list[tuple[str, int, int]] = field(default_factory=list)
    # Gated-novelty under-floor monitoring (pre-registered: exclude+log+monitor).
    neuron_growth_trigger: str = "activation"
    novelty_under_floor_total: int = 0
    novelty_under_floor_by_layer: dict[str, int] = field(default_factory=dict)
    novelty_density_pass_by_layer: dict[str, int] = field(default_factory=dict)
    novelty_ce_gate_pass_by_layer: dict[str, int] = field(default_factory=dict)
    novelty_candidates_by_layer: dict[str, int] = field(default_factory=dict)
    novelty_under_floor_frac_by_layer: dict[str, float] = field(default_factory=dict)
    # Self-calibrating constant snapshots (empty when feature off).
    r_star_live: dict[str, float] = field(default_factory=dict)
    prune_z_cut_live: float | None = None
    straddle_ref_live: dict[str, Any] = field(default_factory=dict)
    self_cal_flags: dict[str, bool] = field(default_factory=dict)
    n_contraction_frac: float = 0.0  # n_contraction / n_active (false-contraction monitor)
    # Neighbor-graph diagnostics (Experiment 1)
    neighbor_metric: str = "grid"
    neighbor_mean_churn: float = 0.0
    neighbor_churn_by_layer: dict[str, float] = field(default_factory=dict)
    neighbor_matched_k_by_layer: dict[str, int] = field(default_factory=dict)
    neighbor_mean_count_by_layer: dict[str, float] = field(default_factory=dict)
    neighbor_unstable: bool = False
    neighbor_signature_ema_mu: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "hidden_counts": dict(self.hidden_counts),
            "edge_counts": self.edge_counts,
            "edges_pruned_init_dense": self.edges_pruned_init_dense,
            "edges_pruned_incentive_grown": self.edges_pruned_incentive_grown,
            "edges_pruned_competitive": self.edges_pruned_competitive,
            "edges_pruned_collateral": self.edges_pruned_collateral,
            "edges_grown": self.edges_grown,
            "edges_grown_incentive": self.edges_grown_incentive,
            "edges_grown_seed": self.edges_grown_seed,
            "neurons_pruned": self.neurons_pruned,
            "neurons_grown": self.neurons_grown,
            "edge_grow_precap": self.edge_grow_precap,
            "edge_missing_in_radius": self.edge_missing_in_radius,
            "edge_candidates_evaluated": self.edge_candidates_evaluated,
            "edge_blocked_cooldown": self.edge_blocked_cooldown,
            "edge_blocked_eligibility": self.edge_blocked_eligibility,
            "edge_growth_no_contraction": self.edge_growth_no_contraction,
            "edge_gnc_on_cooldown": self.edge_gnc_on_cooldown,
            "edge_gnc_off_cooldown": self.edge_gnc_off_cooldown,
            "n_growth_pressure": self.n_growth_pressure,
            "n_contraction_pressure": self.n_contraction_pressure,
            "n_active_hidden": self.n_active_hidden,
            "growth_pressure_by_layer": dict(self.growth_pressure_by_layer),
            "contraction_pressure_by_layer": dict(self.contraction_pressure_by_layer),
            "neuron_grow_precap": self.neuron_grow_precap,
            "edge_grow_postcap_attempts": self.edge_grow_postcap_attempts,
            "neuron_grow_postcap_attempts": self.neuron_grow_postcap_attempts,
            "cap_hits": dict(self.cap_hits),
            "cooldown_blocks": self.cooldown_blocks,
            "same_step_free_reuse_blocked": self.same_step_free_reuse_blocked,
            "n_pairs_growth_no_contraction": len(self.pairs_growth_no_contraction),
            "n_pairs_gnc_on_cooldown": len(self.pairs_gnc_on_cooldown),
            "n_pairs_candidates_evaluated": len(self.pairs_candidates_evaluated),
            "neuron_prune_keys": [list(k) for k in self.neuron_prune_keys],
            "neuron_prune_records": list(self.neuron_prune_records),
            "neuron_grow_keys": [list(k) for k in self.neuron_grow_keys],
            "edge_prune_keys": [list(k) for k in self.edge_prune_keys],
            "edge_grow_keys": [list(k) for k in self.edge_grow_keys],
            "neuron_growth_trigger": self.neuron_growth_trigger,
            "novelty_under_floor_total": self.novelty_under_floor_total,
            "novelty_under_floor_by_layer": dict(self.novelty_under_floor_by_layer),
            "novelty_density_pass_by_layer": dict(self.novelty_density_pass_by_layer),
            "novelty_ce_gate_pass_by_layer": dict(self.novelty_ce_gate_pass_by_layer),
            "novelty_candidates_by_layer": dict(self.novelty_candidates_by_layer),
            "novelty_under_floor_frac_by_layer": dict(
                self.novelty_under_floor_frac_by_layer
            ),
            "r_star_live": dict(self.r_star_live),
            "prune_z_cut_live": self.prune_z_cut_live,
            "straddle_ref_live": dict(self.straddle_ref_live) if self.straddle_ref_live else {},
            "self_cal_flags": dict(self.self_cal_flags),
            "n_contraction_frac": self.n_contraction_frac,
            "neighbor_metric": self.neighbor_metric,
            "neighbor_mean_churn": self.neighbor_mean_churn,
            "neighbor_churn_by_layer": dict(self.neighbor_churn_by_layer),
            "neighbor_matched_k_by_layer": dict(self.neighbor_matched_k_by_layer),
            "neighbor_mean_count_by_layer": dict(self.neighbor_mean_count_by_layer),
            "neighbor_unstable": self.neighbor_unstable,
            "neighbor_signature_ema_mu": self.neighbor_signature_ema_mu,
        }


class PlasticityOrchestrator:
    """Runs one structural update step on a :class:`DynamicMLPReasoningNet`."""

    def __init__(
        self,
        config: PlasticityConfig | None = None,
        *,
        policy: ThresholdPolicy | None = None,
    ) -> None:
        self.config = config or PlasticityConfig()
        self.policy = policy  # built lazily with capacities from net
        self.history: list[StructureEventLog] = []
        # Cumulative survival tracking: created vs pruned by origin.
        self.cumulative_pruned_init = 0
        self.cumulative_pruned_incentive = 0
        self.cumulative_grown_edges = 0
        self.input_signal_ema: Tensor | None = None
        self.input_ema_decay: float = 0.99
        # When True, select_edges_to_grow records pair IDs for persistence traces.
        self.collect_edge_pairs: bool = False
        # Self-calibrating running statistics (optional; warm-start required).
        flags = self.config.self_calibrate_flags()
        self.self_cal = SelfCalibrateState.from_config(
            flags,
            mu_r_star=float(self.config.self_cal_mu_r_star),
            mu_prune=float(self.config.self_cal_mu_prune),
            mu_straddle=float(self.config.self_cal_mu_straddle),
            hybrid_period=self.config.self_cal_hybrid_period,
            prune_target_quantile=float(self.config.self_cal_prune_quantile),
        )
        # Same-layer neighbor graph (grid | feature). Single-variable Experiment 1.
        fixed_k = None
        if self.config.neighbor_top_k is not None:
            k = int(self.config.neighbor_top_k)
            fixed_k = {h: k for h in HIDDEN_LAYERS}
        self.neighbors = NeighborGraph(
            metric=str(self.config.neighbor_metric),
            radius=float(self.config.radius),
            churn_bound=float(self.config.neighbor_churn_bound),
            similarity=str(self.config.neighbor_similarity),
            signature_ema_mu=float(self.config.neighbor_signature_ema_mu),
            fixed_k_by_layer=fixed_k,
        )
        self.neighbor_refresh_log: list[dict[str, Any]] = []

    def ensure_policy(self, net: DynamicMLPReasoningNet) -> ThresholdPolicy:
        if self.policy is not None:
            return self.policy
        caps = {h: int(net.active_mask(h).numel()) for h in HIDDEN_LAYERS}
        self.policy = build_threshold_policy(
            self.config.threshold_kind,
            tau=self.config.tau,
            target_rates=self.config.resolved_target_rates(),
            mu=self.config.mu,
            capacities=caps,
            pressure_margin=self.config.pressure_margin,
        )
        return self.policy

    def warm_start_self_cal(
        self,
        *,
        target_rates: dict[str, float] | None = None,
        prune_z_cut: float = -1.0,
    ) -> None:
        """Initialize self-cal trackers from a probe (required when flags ON).

        ``target_rates`` must come from :func:`probe_target_rates` (or equivalent
        one-epoch ActivationMagnitude means) — never zeros or a foreign dataset.
        """
        if self.self_cal.flags.r_star:
            if not target_rates:
                raise ValueError("warm_start_self_cal requires target_rates when r_star ON")
            assert self.self_cal.rates is not None
            self.self_cal.rates.warm_start(target_rates)
            self.config.target_rates = {
                h: float(target_rates[h]) for h in HIDDEN_LAYERS
            }
            if self.policy is not None:
                self.policy.target_rates = dict(self.config.target_rates)
        if self.self_cal.flags.prune_cutoff:
            assert self.self_cal.prune is not None
            self.self_cal.prune.warm_start(float(prune_z_cut))
        if self.self_cal.flags.straddle and self.self_cal.straddle is not None:
            self.self_cal.straddle.warm_start()

    @torch.no_grad()
    def update_input_signal_ema(self, act_in: Tensor) -> None:
        """EMA of mean |input| per dim from a batch ``[B, input_dim]``."""
        stat = act_in.detach().float().abs().mean(dim=0).cpu()
        if self.input_signal_ema is None:
            self.input_signal_ema = stat.clone()
        else:
            d = self.input_ema_decay
            self.input_signal_ema.mul_(d).add_(stat, alpha=1.0 - d)

    def _signals(
        self,
        net: DynamicMLPReasoningNet,
        neuron_scores: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        scores = dict(neuron_scores)
        if self.input_signal_ema is not None:
            scores["in"] = self.input_signal_ema
        return population_signals_from_activation(net, scores)

    def refresh_neighbors(
        self,
        net: DynamicMLPReasoningNet,
        layer_acts: dict[str, Tensor] | None = None,
        *,
        apply_stability_response: bool = True,
    ) -> dict[str, Any]:
        """Recompute same-layer neighbor masks (once per plasticity cadence).

        For ``neighbor_metric=feature``, ``layer_acts`` (probe post-acts) is
        required. On ``NEIGHBORHOOD_UNSTABLE``, applies the pre-committed
        EMA-signature response and refreshes once more when enabled.
        """
        res = self.neighbors.refresh(net, layer_acts=layer_acts)
        if res.unstable and apply_stability_response:
            self.neighbors.apply_stability_response()
            # Re-refresh with EMA so the step does not run on an unstable graph.
            res = self.neighbors.refresh(net, layer_acts=layer_acts)
            res.notes.append("stability_response_applied_ema_re_refresh")
        d = res.as_dict()
        self.neighbor_refresh_log.append(d)
        return d

    @torch.no_grad()
    def step(
        self,
        net: DynamicMLPReasoningNet,
        neuron_scores: dict[str, Tensor],
        *,
        optimizer: Optimizer | None = None,
        epoch: int | None = None,
        signal: Any = None,
        gated_novelty: GatedNoveltyContext | None = None,
    ) -> StructureEventLog:
        """One plasticity evaluation: prune then grow (with cooldown).

        ``signal``: optional ActivationMagnitude (or compatible) for seeding
        newborn neuron EMAs after growth.

        ``gated_novelty``: required when
        ``config.neuron_growth_trigger == "gated_novelty"`` — probe-time scores
        for the three-part rule (novelty ∧ CE gate ∧ density floor). Units with
        ``|N| < k_min`` are excluded and counted on the log (under-floor monitor).
        """
        policy = self.ensure_policy(net)
        blocks_before = net.cooldown.same_step_blocks
        step_idx = net.begin_plasticity_step()
        # Decay temporary birth biases with age (pre-registered schedule).
        net.update_birth_biases(self.config.newborn_birth_bias_steps)
        signals = self._signals(net, neuron_scores)

        # Map fc* neuron scores → h* for threshold update.
        hidden_signals = {
            "h1": signals["h1"],
            "h2": signals["h2"],
            "h3": signals["h3"],
        }
        active = {h: net.active_mask(h).cpu() for h in HIDDEN_LAYERS}
        cfg = self.config
        log = StructureEventLog(step=step_idx, epoch=epoch)
        log.self_cal_flags = dict(cfg.self_calibrate_flags())
        log.neighbor_metric = str(cfg.neighbor_metric)
        if self.neighbors.last_result is not None:
            lr = self.neighbors.last_result
            log.neighbor_mean_churn = float(lr.mean_churn)
            log.neighbor_churn_by_layer = dict(lr.churn_by_layer)
            log.neighbor_matched_k_by_layer = dict(lr.matched_k_by_layer)
            log.neighbor_mean_count_by_layer = dict(lr.mean_count_by_layer)
            log.neighbor_unstable = bool(lr.unstable or self.neighbors.unstable_ever)
            log.neighbor_signature_ema_mu = float(self.neighbors.signature_ema_mu)
        else:
            log.neighbor_signature_ema_mu = float(self.neighbors.signature_ema_mu)

        # --- Self-cal r*_ℓ (before pressure decisions) ---
        if self.self_cal.flags.r_star:
            if self.self_cal.rates is None or not self.self_cal.rates.warmed:
                raise RuntimeError(
                    "self_calibrate.r_star is ON but LayerRateTracker was not "
                    "warm-started; call orch.warm_start_self_cal(...) after the "
                    "one-epoch activation probe"
                )
            live_rates = self.self_cal.rates.update(hidden_signals, active)
            # Keep policy + config in sync (contraction, newborn EMA seed).
            policy.target_rates = dict(live_rates)
            if hasattr(policy, "target_rate"):
                policy.target_rate = sum(live_rates.values()) / len(live_rates)
            self.config.target_rates = dict(live_rates)
            log.r_star_live = dict(live_rates)

        policy.update(hidden_signals, active)

        trigger_mode = str(cfg.neuron_growth_trigger).strip().lower()
        log.neuron_growth_trigger = trigger_mode
        use_gated = trigger_mode in ("gated_novelty", "novelty", "gated")
        if use_gated and gated_novelty is None:
            raise ValueError(
                "neuron_growth_trigger='gated_novelty' requires gated_novelty "
                "context (probe CE + activations); under-floor exclusion is "
                "explicit and must not silently fall back to act>r*"
            )

        # Marginal pressure counts (before structure edits) — for scope wording
        # (growth rare vs growth/contraction co-location on pairs).
        for h in HIDDEN_LAYERS:
            a = active[h]
            g = policy.growth_pressure(h).cpu() & a
            c = policy.contraction_pressure(h).cpu() & a
            ng = int(g.sum())
            nc = int(c.sum())
            log.growth_pressure_by_layer[h] = ng
            log.contraction_pressure_by_layer[h] = nc
            log.n_growth_pressure += ng
            log.n_contraction_pressure += nc
            log.n_active_hidden += int(a.sum())
        if log.n_active_hidden > 0:
            log.n_contraction_frac = float(log.n_contraction_pressure) / float(
                log.n_active_hidden
            )

        # --- (2) competitive edge prune (z-score / fraction; does not read τ) ---
        for layer in LINEAR_LAYERS:
            z = competitive_edge_zscores(net, layer, signals, cfg.radius)
            to_drop = select_edges_to_prune(net, layer, z, cfg.prune_edge_fraction)
            for oi, ii in to_drop:
                try:
                    ev = net.remove_edge(layer, oi, ii)
                except ValueError:
                    continue
                log.edges_pruned_competitive += 1
                log.edge_prune_keys.append((layer, oi, ii))
                if ev["origin"] == EDGE_ORIGIN_INIT:
                    log.edges_pruned_init_dense += 1
                    self.cumulative_pruned_init += 1
                elif ev["origin"] in (EDGE_ORIGIN_INCENTIVE, EDGE_ORIGIN_GAP_IMPRINT):
                    # Gap-seeded edges count as grown structure for prune stats.
                    log.edges_pruned_incentive_grown += 1
                    self.cumulative_pruned_incentive += 1

        # Neuron z-scores once (neighborhood-local when neighbor graph is ready);
        # optional self-cal prune cut.
        neuron_z: dict[str, Tensor] = {}
        for h in HIDDEN_LAYERS:
            nmask = self.neighbors.get_mask(h)
            neuron_z[h] = competitive_neuron_zscores(
                net, h, signals, neighbor_mask=nmask
            )
        z_cut = float(cfg.prune_neuron_z)
        if self.self_cal.flags.prune_cutoff:
            if self.self_cal.prune is None or not self.self_cal.prune.warmed:
                raise RuntimeError(
                    "self_calibrate.prune_cutoff is ON but PruneCutoffTracker "
                    "was not warm-started"
                )
            z_cut = float(
                self.self_cal.prune.update(neuron_z, active)
            )
            log.prune_z_cut_live = z_cut
        else:
            log.prune_z_cut_live = z_cut

        # --- prune neurons (contraction pressure uses τ for Variant A) ---
        for h in HIDDEN_LAYERS:
            z = neuron_z[h]
            for slot in select_neurons_to_prune(
                net,
                h,
                z,
                policy,
                z_cut,
                grace_steps=cfg.newborn_grace_steps,
            ):
                try:
                    ev = net.deactivate_neuron(h, slot)
                except ValueError:
                    continue
                log.neurons_pruned += 1
                log.neuron_prune_keys.append((h, int(slot)))
                log.neuron_prune_records.append(
                    {
                        "layer": h,
                        "slot": int(slot),
                        "x": ev.get("x"),
                        "y": ev.get("y"),
                        "neuron_origin": ev.get("neuron_origin"),
                        "birth_kind": ev.get("birth_kind"),
                        "parent_slot": ev.get("parent_slot"),
                        "birth_step": ev.get("birth_step"),
                    }
                )
                for lin, o, i, origin in ev["edges_removed"]:
                    log.edges_pruned_collateral += 1
                    log.edge_prune_keys.append((lin, int(o), int(i)))
                    if origin == EDGE_ORIGIN_INIT:
                        log.edges_pruned_init_dense += 1
                        self.cumulative_pruned_init += 1
                    elif origin in (EDGE_ORIGIN_INCENTIVE, EDGE_ORIGIN_GAP_IMPRINT):
                        log.edges_pruned_incentive_grown += 1
                        self.cumulative_pruned_incentive += 1

        # --- (4) grow edges (incentive path; threshold uses τ / max(θ_g)) ---
        for layer in LINEAR_LAYERS:
            inc = edge_incentive_matrix(net, layer, signals, cfg.radius)
            picks, estats = select_edges_to_grow(
                net,
                layer,
                inc,
                policy,
                cfg.max_new_edges_per_layer,
                eligibility_mode=cfg.edge_eligibility_mode,
                return_stats=True,
                collect_pairs=self.collect_edge_pairs,
            )
            log.edge_missing_in_radius += int(estats.missing_in_radius)
            log.edge_candidates_evaluated += int(estats.candidates_evaluated)
            log.edge_blocked_cooldown += int(estats.blocked_cooldown)
            log.edge_blocked_eligibility += int(estats.blocked_eligibility)
            log.edge_growth_no_contraction += int(estats.growth_no_contraction)
            log.edge_gnc_on_cooldown += int(estats.growth_no_contraction_on_cooldown)
            log.edge_gnc_off_cooldown += int(estats.growth_no_contraction_off_cooldown)
            log.edge_grow_precap += int(estats.threshold_passes)
            log.edge_grow_postcap_attempts += len(picks)
            if self.collect_edge_pairs:
                log.pairs_growth_no_contraction.extend(
                    estats.pairs_growth_no_contraction
                )
                log.pairs_gnc_on_cooldown.extend(
                    estats.pairs_growth_no_contraction_on_cooldown
                )
                log.pairs_candidates_evaluated.extend(
                    estats.pairs_candidates_evaluated
                )
            for oi, ii, _score in picks:
                try:
                    net.add_edge(
                        layer,
                        oi,
                        ii,
                        origin=EDGE_ORIGIN_INCENTIVE,
                        optimizer=optimizer,
                    )
                    log.edges_grown += 1
                    log.edges_grown_incentive += 1
                    log.edge_grow_keys.append((layer, oi, ii))
                    self.cumulative_grown_edges += 1
                except (ValueError, RuntimeError):
                    continue

        # --- (5) grow neurons ---
        for h in HIDDEN_LAYERS:
            log.cap_hits[h] = net.at_capacity(h)
            if use_gated:
                assert gated_novelty is not None
                st = gated_novelty.stats_for(h)
                n_act = max(int(st.n_active), 1)
                log.novelty_under_floor_by_layer[h] = int(st.n_under_floor)
                log.novelty_density_pass_by_layer[h] = int(st.n_density_pass)
                log.novelty_ce_gate_pass_by_layer[h] = int(st.n_ce_gate_pass)
                log.novelty_candidates_by_layer[h] = int(st.n_candidates)
                log.novelty_under_floor_frac_by_layer[h] = float(
                    st.n_under_floor
                ) / float(n_act)
                log.novelty_under_floor_total += int(st.n_under_floor)
                triggers, precap_n = select_neurons_to_grow_gated_novelty(
                    net,
                    h,
                    list(st.candidates),
                    cfg.max_new_neurons_per_layer,
                    rank_fraction=float(cfg.novelty_rank_fraction),
                    return_precap_count=True,
                )
            else:
                triggers, precap_n = select_neurons_to_grow(
                    net,
                    h,
                    policy,
                    cfg.max_new_neurons_per_layer,
                    return_precap_count=True,
                )
            log.neuron_grow_precap += int(precap_n)
            log.neuron_grow_postcap_attempts += len(triggers)
            for trigger in triggers:
                if net.at_capacity(h):
                    log.cap_hits[h] = True
                    break
                winit = str(self.config.newborn_weight_init).strip().lower()
                gap_row = None
                if winit in ("gap_imprinted", "gap_imprint", "gap"):
                    if gated_novelty is None:
                        raise ValueError(
                            "gap_imprinted birth requires gated_novelty context "
                            "with pre_acts and gap_masks"
                        )
                    gap_row = self._build_gap_imprint_row(
                        gated_novelty, h, int(trigger)
                    )
                # gap_imprinted: do NOT use the small decaying birth_bias scaffold
                # (0.5) — realized pre sits near −10; scaffold is ~20× too small.
                # Operating-point bias is set from probe stats after edge realization.
                bb = (
                    0.0
                    if winit in ("gap_imprinted", "gap_imprint", "gap")
                    else float(self.config.newborn_birth_bias)
                )
                try:
                    info = net.activate_neuron(
                        h,
                        slot=None,
                        origin=NEURON_ORIGIN_GROWN,
                        optimizer=optimizer,
                        birth_bias=bb,
                        parent_slot=int(trigger),
                        weight_init=winit,
                        parent_noise=float(self.config.newborn_parent_noise),
                        copy_outgoing=bool(self.config.newborn_copy_outgoing),
                        parent_coord_jitter=float(
                            self.config.newborn_parent_coord_jitter
                        ),
                        gap_weight_row=gap_row,
                    )
                except (RuntimeError, ValueError):
                    # Fall back to kaiming if parent/gap seed fails.
                    try:
                        info = net.activate_neuron(
                            h,
                            slot=None,
                            origin=NEURON_ORIGIN_GROWN,
                            optimizer=optimizer,
                            birth_bias=float(self.config.newborn_birth_bias),
                            weight_init="kaiming",
                        )
                    except RuntimeError:
                        break
                log.neurons_grown += 1
                log.neuron_grow_keys.append((h, int(info["slot"])))
                if isinstance(policy, SlidingThreshold):
                    policy.reset_slot(h, info["slot"])
                self._seed_newborn_signal(signal, h, int(info["slot"]))
                # Parent-perturbed already copies the parent's edge fan.
                # gap_imprinted and kaiming need incentive seed-edge bootstrap.
                # Inject newborn's warm-start score into the *local* signals map
                # used for seed incentive — otherwise s_post stays 0 (pre-birth
                # snapshot) and no seed edges form, leaving only birth_bias.
                winit_done = str(info.get("weight_init", "kaiming"))
                if winit_done == "parent_perturbed":
                    pass  # edge fan already copied from trigger
                elif winit_done == "gap_imprinted":
                    # Open by *separating power*, not |W|; then rescale to open fan-in;
                    # then set Parameter bias from realized pre (straddle midpoint).
                    assert gated_novelty is not None
                    self._seed_edges_gap_imprint(
                        net,
                        h,
                        int(info["slot"]),
                        int(trigger),
                        gated_novelty,
                        optimizer,
                        log,
                    )
                    self._set_gap_imprint_operating_bias(
                        net,
                        h,
                        int(info["slot"]),
                        int(trigger),
                        gated_novelty,
                    )
                else:
                    # kaiming: incentive seed with warm-started post score
                    seed_signals = dict(signals)
                    rates = self.config.resolved_target_rates()
                    s_new = seed_signals[h].clone()
                    s_new[int(info["slot"])] = float(rates[h])
                    seed_signals[h] = s_new
                    self._seed_edges_for_neuron(
                        net,
                        h,
                        info["slot"],
                        seed_signals,
                        policy,
                        optimizer,
                        log,
                    )

        log.hidden_counts = net.hidden_counts()
        log.edge_counts = net.edge_counts(by_origin=True)
        log.cooldown_blocks = net.cooldown.same_step_blocks - blocks_before
        log.same_step_free_reuse_blocked = log.cooldown_blocks
        self.history.append(log)
        return log

    def _build_gap_imprint_row(
        self,
        gated_novelty: GatedNoveltyContext,
        layer: str,
        trigger: int,
    ) -> Tensor:
        """Residual pre-act direction on the trigger's CE-excess gap samples."""
        key = (layer, int(trigger))
        if key not in gated_novelty.gap_masks:
            raise ValueError(
                f"no gap_mask for {layer}[{trigger}]; cannot gap-imprint"
            )
        if layer not in gated_novelty.pre_acts:
            raise ValueError(
                f"no pre_acts for {layer} in gated_novelty context; "
                "collect_gated_novelty_probe must record incoming pre-activations"
            )
        pre = gated_novelty.pre_acts[layer]
        mask = gated_novelty.gap_masks[key]
        ce = gated_novelty.ce
        direction = gap_imprint_direction(
            pre, mask, ce, subtract_global_mean=True
        )
        return scale_direction_to_kaiming_relu(direction)

    def _seed_newborn_signal(
        self,
        signal: Any,
        hidden_layer: str,
        slot: int,
    ) -> None:
        """Seed activation EMA for a grown slot (pre-registered warm-start)."""
        if signal is None:
            return
        mode = str(self.config.newborn_ema_init).strip().lower()
        if mode in ("zero", "none", "off"):
            return
        # Map hidden population → fc* postsynaptic score buffer.
        fc = {"h1": "fc1", "h2": "fc2", "h3": "fc3"}[hidden_layer]
        rates = self.config.resolved_target_rates()
        value = float(rates[hidden_layer])
        neuron = getattr(signal, "_neuron", None)
        if not isinstance(neuron, dict) or fc not in neuron:
            return
        buf = neuron[fc]
        if slot < 0 or slot >= buf.shape[0]:
            return
        buf[slot] = value

    def _seed_edges_gap_imprint(
        self,
        net: DynamicMLPReasoningNet,
        hidden_layer: str,
        slot: int,
        trigger: int,
        gated_novelty: GatedNoveltyContext,
        optimizer: Optimizer | None,
        log: StructureEventLog,
    ) -> None:
        """Realize gap residual: open by *separating power*, rescale to open fan-in.

        **Why not top-|W|:** magnitude picks large residual coefficients, which
        can be high-dynamic-range dims that are not diagnostic of gap vs
        non-gap. Selection uses per-dim
        ``|mean_gap - mean_nongap| / std`` (aligned with residual sign).

        After opening, rewrite open incoming weights from the residual
        restricted to those dims and scale with Kaiming fan_in = n_open
        (not full ambient fan-in).
        """
        cfg = self.config
        k = max(int(cfg.seed_edges_per_new_neuron), 1)
        k_in = max(k, 16)
        k_out = max(k, 8)
        radius = float(cfg.radius)
        post_c = net.coords(hidden_layer).cpu().float()[slot]

        key = (hidden_layer, int(trigger))
        if key not in gated_novelty.gap_masks or hidden_layer not in gated_novelty.pre_acts:
            # Fall back: no probe tensors — cannot score separation.
            return
        pre_acts = gated_novelty.pre_acts[hidden_layer]
        gap_mask = gated_novelty.gap_masks[key]
        residual = gap_imprint_direction(
            pre_acts, gap_mask, gated_novelty.ce, subtract_global_mean=True
        )
        sep_scores = per_dimension_separation_scores(
            pre_acts, gap_mask, residual=residual
        )

        for layer, (pre, post) in _LINEAR_ENDPOINTS.items():
            if post != hidden_layer and pre != hidden_layer:
                continue
            mod = net.linear_module(layer)
            em = net.edge_mask(layer)
            candidates: list[tuple[int, int, float]] = []
            if post == hidden_layer:
                # Incoming: rank by separating power, not |W|.
                pre_a = net.population_active(pre).cpu()
                pre_c = net.coords(pre).cpu().float()
                scores = sep_scores.cpu()
                for j in range(int(scores.numel())):
                    if not bool(pre_a[j]):
                        continue
                    if bool(em[slot, j]):
                        continue
                    if not net.cooldown.edge_eligible(layer, slot, j):
                        continue
                    d = float(torch.linalg.norm(post_c - pre_c[j]).item())
                    if pre != "in" and d > radius:
                        continue
                    score = float(scores[j].item())
                    if score <= 0.0:
                        continue
                    candidates.append((slot, j, score))
                candidates.sort(key=lambda t: t[2], reverse=True)
                picks = candidates[:k_in]
            else:
                # Outgoing: still local top-|w| (kaiming col; no gap residual).
                post_a = net.population_active(post).cpu()
                post_coords = net.coords(post).cpu().float()
                wcol = mod.weight[:, slot].detach().float().cpu().abs()
                pre_c = net.coords(hidden_layer).cpu().float()[slot]
                for i in range(wcol.numel()):
                    if not bool(post_a[i]):
                        continue
                    if bool(em[i, slot]):
                        continue
                    if not net.cooldown.edge_eligible(layer, i, slot):
                        continue
                    d = float(torch.linalg.norm(post_coords[i] - pre_c).item())
                    if d > radius and post in HIDDEN_LAYERS:
                        continue
                    candidates.append((i, slot, float(wcol[i].item())))
                candidates.sort(key=lambda t: t[2], reverse=True)
                picks = candidates[:k_out]

            for oi, ii, _v in picks:
                try:
                    net.add_edge(
                        layer,
                        oi,
                        ii,
                        origin=EDGE_ORIGIN_GAP_IMPRINT,
                        optimizer=optimizer,
                    )
                    log.edges_grown += 1
                    log.edges_grown_seed += 1
                    log.edge_grow_keys.append((layer, oi, ii))
                    self.cumulative_grown_edges += 1
                except (ValueError, RuntimeError):
                    continue

        # Rescale *incoming* open weights: residual on open support, fan_in=n_open.
        self._rescale_gap_imprint_open_row(
            net, hidden_layer, slot, residual
        )

    def _rescale_gap_imprint_open_row(
        self,
        net: DynamicMLPReasoningNet,
        hidden_layer: str,
        slot: int,
        residual: Tensor,
    ) -> None:
        """Set open incoming weights to residual|_{open}, Kaiming-scaled by n_open."""
        lin = {"h1": "fc1", "h2": "fc2", "h3": "fc3"}[hidden_layer]
        mod = net.linear_module(lin)
        em = net.edge_mask(lin)[slot].detach().cpu().bool()
        n_open = int(em.sum().item())
        if n_open <= 0:
            return
        r = residual.detach().float().reshape(-1).cpu()
        if r.numel() != em.numel():
            return
        open_r = r[em]
        scaled_open = scale_direction_to_kaiming_relu(open_r, fan_in=n_open)
        with torch.no_grad():
            full = torch.zeros(r.numel(), dtype=torch.float32)
            full[em] = scaled_open.float()
            mod.weight[slot].copy_(
                full.to(device=mod.weight.device, dtype=mod.weight.dtype)
            )
            if mod.bias is not None:
                mod.bias[slot] = 0.0

    def _set_gap_imprint_operating_bias(
        self,
        net: DynamicMLPReasoningNet,
        hidden_layer: str,
        slot: int,
        trigger: int,
        gated_novelty: GatedNoveltyContext,
        *,
        mode: str = "straddle",
    ) -> float:
        """Set Parameter bias so realized pre straddles ReLU zero on gap vs non-gap.

        Derived from measurement (not scaffold 0.5): after sep-power edges + open-fan
        scale, mean pre sits near −10 with d≈1 separation — both clouds deep in the
        ReLU dead zone. Pure weight rescale cannot flip a negative mean.

        ``straddle`` (default): B = −(mean_pre_gap + mean_pre_nongap) / 2
        so the midpoint between clouds lands at 0; expected mean_gap → +Δ/2,
        mean_nongap → −Δ/2. Uses the same probe pre_acts + birth gap_mask.
        """
        key = (hidden_layer, int(trigger))
        if key not in gated_novelty.gap_masks or hidden_layer not in gated_novelty.pre_acts:
            return 0.0
        lin = {"h1": "fc1", "h2": "fc2", "h3": "fc3"}[hidden_layer]
        mod = net.linear_module(lin)
        if mod.bias is None:
            return 0.0
        pre = gated_novelty.pre_acts[hidden_layer].detach().float().cpu()  # [B, fan]
        m = gated_novelty.gap_masks[key].bool().cpu()
        n = min(pre.shape[0], m.numel())
        pre, m = pre[:n], m[:n]
        w = mod.weight[slot].detach().float().cpu()
        em = net.edge_mask(lin)[slot].detach().cpu().bool()
        w_eff = w * em.float()
        # Per-sample pre-activation = W_eff · pre_syn  (bias still 0 here)
        unit_pre = (pre * w_eff.unsqueeze(0)).sum(dim=1)
        if not bool(m.any()) or not bool((~m).any()):
            return 0.0
        mean_gap = float(unit_pre[m].mean().item())
        mean_nongap = float(unit_pre[~m].mean().item())
        mean_whole = float(unit_pre.mean().item())
        if mode == "gap_zero":
            # Minimal: push gap mean just above 0
            B = -mean_gap + 1e-3
        elif mode == "center_whole":
            B = -mean_whole
        else:
            # straddle midpoint between gap and non-gap clouds (live primary)
            B = -0.5 * (mean_gap + mean_nongap)
            # Optional self-cal layer reference update / degenerate fallback
            if self.self_cal.flags.straddle and self.self_cal.straddle is not None:
                self.self_cal.straddle.update_from_birth(
                    hidden_layer, mean_gap, mean_nongap
                )
        with torch.no_grad():
            mod.bias[slot] = float(B)
        return float(B)

    def _seed_edges_for_neuron(
        self,
        net: DynamicMLPReasoningNet,
        hidden_layer: str,
        slot: int,
        signals: dict[str, Tensor],
        policy: ThresholdPolicy,
        optimizer: Optimizer | None,
        log: StructureEventLog,
    ) -> None:
        """Connect a newly grown neuron to up to K local partners via incentive."""
        cfg = self.config
        k = cfg.seed_edges_per_new_neuron
        for layer, (pre, post) in _LINEAR_ENDPOINTS.items():
            if post != hidden_layer and pre != hidden_layer:
                continue
            inc = edge_incentive_matrix(net, layer, signals, cfg.radius)
            # Force growth pressure consideration: temporarily the new unit may
            # not yet show growth in policy; seed using pure incentive rank
            # within R among missing edges incident to this slot.
            em = net.edge_mask(layer).cpu()
            candidates: list[tuple[int, int, float]] = []
            if post == hidden_layer:
                row = inc[slot]
                for j in range(row.numel()):
                    if bool(em[slot, j]):
                        continue
                    if not bool(net.population_active(pre)[j]):
                        continue
                    if not net.cooldown.edge_eligible(layer, slot, j):
                        continue
                    val = float(row[j].item())
                    if val > 0:
                        candidates.append((slot, j, val))
            if pre == hidden_layer:
                col = inc[:, slot]
                for i in range(col.numel()):
                    if bool(em[i, slot]):
                        continue
                    if not bool(net.population_active(post)[i]):
                        continue
                    if not net.cooldown.edge_eligible(layer, i, slot):
                        continue
                    val = float(col[i].item())
                    if val > 0:
                        candidates.append((i, slot, val))
            candidates.sort(key=lambda t: t[2], reverse=True)
            for oi, ii, _v in candidates[:k]:
                try:
                    net.add_edge(
                        layer,
                        oi,
                        ii,
                        origin=EDGE_ORIGIN_INCENTIVE,
                        optimizer=optimizer,
                    )
                    log.edges_grown += 1
                    log.edges_grown_seed += 1
                    log.edge_grow_keys.append((layer, oi, ii))
                    self.cumulative_grown_edges += 1
                except (ValueError, RuntimeError):
                    continue
