"""Local self-calibrating running statistics for orchestrator constants.

**Product ship (validated):** ``r*_ℓ`` only — enable ``self_calibrate.r_star``.

**Not shipped:** ``prune_cutoff`` and ``straddle`` trackers remain in-tree for
research flags only. §4.5 ablation: r*_ℓ alone recovers Fashion; prune/straddle
self-cal add no measurable recovery. Pre-registered rule: report optional, do
not ship. Do **not** hybrid-stabilize prune-cutoff — that would stabilize a
conversion that did not earn its place. Product prune uses frozen
``prune_neuron_z=-1.0``; straddle B remains live per-birth (already local).

Warm-start from a one-epoch probe is required for r*_ℓ — never seed from zero.
No outer search, no global objective, no cross-layer coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch
from torch import Tensor

from neuronca.models.dynamic_mlp import HIDDEN_LAYERS


@dataclass
class SelfCalibrateFlags:
    """Independent toggles for each converted constant.

    Product: set ``r_star=True`` only. ``prune_cutoff`` / ``straddle`` are
    research flags (not shipped — see module docstring / §4.5).
    """

    r_star: bool = False
    prune_cutoff: bool = False  # NOT SHIPPED — frozen prune_neuron_z in product
    straddle: bool = False  # NOT SHIPPED — live per-birth B already local

    @classmethod
    def from_mapping(cls, m: Mapping[str, object] | None) -> SelfCalibrateFlags:
        if not m:
            return cls()
        return cls(
            r_star=bool(m.get("r_star", False)),
            prune_cutoff=bool(m.get("prune_cutoff", False)),
            straddle=bool(m.get("straddle", False)),
        )

    def any_enabled(self) -> bool:
        return self.r_star or self.prune_cutoff or self.straddle

    def as_dict(self) -> dict[str, bool]:
        return {
            "r_star": self.r_star,
            "prune_cutoff": self.prune_cutoff,
            "straddle": self.straddle,
        }


@dataclass
class LayerRateTracker:
    """Per-layer homeostatic target r*_ℓ as EMA of mean active activation.

    ``r*_ℓ(t) = μ · r*_ℓ(t-1) + (1-μ) · mean_act_active(t)``
    """

    mu: float = 0.999
    rates: dict[str, float] = field(default_factory=dict)
    _warmed: bool = False
    # Hybrid: only update every ``hybrid_period`` steps (None = every step).
    hybrid_period: int | None = None
    _steps: int = 0
    history: list[dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 < float(self.mu) < 1.0:
            raise ValueError(f"mu must be in (0,1), got {self.mu}")
        if self.hybrid_period is not None and int(self.hybrid_period) < 1:
            raise ValueError("hybrid_period must be >= 1 or None")

    def warm_start(self, rates: Mapping[str, float]) -> None:
        """Initialize from probe means — required before first update."""
        out: dict[str, float] = {}
        for h in HIDDEN_LAYERS:
            if h not in rates:
                raise ValueError(f"warm_start missing layer {h}")
            v = float(rates[h])
            if v <= 0.0:
                raise ValueError(
                    f"warm_start r* for {h} must be > 0 (got {v}); "
                    "never seed from zero"
                )
            out[h] = v
        self.rates = out
        self._warmed = True
        self._steps = 0
        self.history = [dict(out)]

    @property
    def warmed(self) -> bool:
        return self._warmed

    def snapshot(self) -> dict[str, float]:
        return {h: float(self.rates[h]) for h in HIDDEN_LAYERS if h in self.rates}

    def update(
        self,
        neuron_signals: Mapping[str, Tensor],
        active: Mapping[str, Tensor],
        *,
        force: bool = False,
    ) -> dict[str, float]:
        """Update rates from current EMAs; return live rates."""
        if not self._warmed:
            raise RuntimeError(
                "LayerRateTracker.update called before warm_start; "
                "run the one-epoch probe first"
            )
        self._steps += 1
        period = self.hybrid_period
        should = force or period is None or (self._steps % int(period) == 0)
        if should:
            mu = float(self.mu)
            for h in HIDDEN_LAYERS:
                s = neuron_signals[h].detach().float().cpu().reshape(-1)
                a = active[h].detach().cpu().bool().reshape(-1)
                n = min(s.numel(), a.numel())
                s, a = s[:n], a[:n]
                if not bool(a.any()):
                    continue
                mean_act = float(s[a].mean().item())
                prev = float(self.rates[h])
                self.rates[h] = mu * prev + (1.0 - mu) * mean_act
            self.history.append(self.snapshot())
        return self.snapshot()


@dataclass
class PruneCutoffTracker:
    """Running α-quantile of active neuron z-scores → live prune z-cut.

    Under normality, α≈0.1587 recovers z≈−1 (product default). Updated as an
    EMA of the empirical quantile each plasticity step (layer-pooled or per-layer
    mean of per-layer quantiles).
    """

    mu: float = 0.999
    target_quantile: float = 0.1587
    z_cut: float = -1.0
    _warmed: bool = False
    hybrid_period: int | None = None
    _steps: int = 0
    history: list[float] = field(default_factory=list)
    # Optional per-layer cuts (mean still stored in z_cut for simple callers).
    z_cut_by_layer: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.mu) < 1.0:
            raise ValueError(f"mu must be in [0,1), got {self.mu}")
        q = float(self.target_quantile)
        if not 0.0 < q < 1.0:
            raise ValueError(f"target_quantile must be in (0,1), got {q}")

    def warm_start(self, z_cut: float = -1.0) -> None:
        self.z_cut = float(z_cut)
        self.z_cut_by_layer = {h: float(z_cut) for h in HIDDEN_LAYERS}
        self._warmed = True
        self._steps = 0
        self.history = [float(z_cut)]

    def warm_start_from_zscores(
        self,
        z_by_layer: Mapping[str, Tensor],
        active: Mapping[str, Tensor],
    ) -> None:
        """Initialize cut from one snapshot of active z-scores."""
        q = self._empirical_quantile(z_by_layer, active)
        self.warm_start(q if q == q else -1.0)

    @property
    def warmed(self) -> bool:
        return self._warmed

    def _empirical_quantile(
        self,
        z_by_layer: Mapping[str, Tensor],
        active: Mapping[str, Tensor],
    ) -> float:
        chunks: list[Tensor] = []
        per: dict[str, float] = {}
        alpha = float(self.target_quantile)
        for h in HIDDEN_LAYERS:
            if h not in z_by_layer or h not in active:
                continue
            z = z_by_layer[h].detach().float().cpu().reshape(-1)
            a = active[h].detach().cpu().bool().reshape(-1)
            n = min(z.numel(), a.numel())
            z, a = z[:n], a[:n]
            if not bool(a.any()):
                continue
            za = z[a]
            qh = float(torch.quantile(za, alpha).item())
            per[h] = qh
            chunks.append(za)
        self.z_cut_by_layer = {
            h: per.get(h, self.z_cut) for h in HIDDEN_LAYERS
        }
        if not chunks:
            return float("nan")
        all_z = torch.cat(chunks, dim=0)
        return float(torch.quantile(all_z, alpha).item())

    def update(
        self,
        z_by_layer: Mapping[str, Tensor],
        active: Mapping[str, Tensor],
        *,
        force: bool = False,
    ) -> float:
        if not self._warmed:
            raise RuntimeError("PruneCutoffTracker.update before warm_start")
        self._steps += 1
        period = self.hybrid_period
        should = force or period is None or (self._steps % int(period) == 0)
        if should:
            q = self._empirical_quantile(z_by_layer, active)
            if q == q:
                mu = float(self.mu)
                self.z_cut = mu * float(self.z_cut) + (1.0 - mu) * q
                self.history.append(float(self.z_cut))
        return float(self.z_cut)


@dataclass
class StraddleRefTracker:
    """Per-layer EMA of gap/non-gap pre-activation means from births.

    Live birth-time (μ_gap, μ_nongap) remains primary for B; this tracker is a
    layer reference for diagnostics / degenerate-mask fallback.
    """

    mu: float = 0.99
    mean_gap: dict[str, float] = field(default_factory=dict)
    mean_nongap: dict[str, float] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)
    _warmed: bool = False
    history: list[dict[str, dict[str, float]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 < float(self.mu) < 1.0:
            raise ValueError(f"mu must be in (0,1), got {self.mu}")

    def warm_start(
        self,
        mean_gap: Mapping[str, float] | None = None,
        mean_nongap: Mapping[str, float] | None = None,
    ) -> None:
        """Optional layer seeds; empty means first birth initializes."""
        self.mean_gap = {h: float(mean_gap[h]) for h in HIDDEN_LAYERS if mean_gap and h in mean_gap}
        self.mean_nongap = {
            h: float(mean_nongap[h]) for h in HIDDEN_LAYERS if mean_nongap and h in mean_nongap
        }
        self._counts = {h: 1 if h in self.mean_gap else 0 for h in HIDDEN_LAYERS}
        self._warmed = True

    @property
    def warmed(self) -> bool:
        return self._warmed

    def update_from_birth(
        self, layer: str, mean_gap: float, mean_nongap: float
    ) -> None:
        if layer not in HIDDEN_LAYERS:
            return
        if not self._warmed:
            self.warm_start()
        mu = float(self.mu)
        n = int(self._counts.get(layer, 0))
        if n <= 0:
            self.mean_gap[layer] = float(mean_gap)
            self.mean_nongap[layer] = float(mean_nongap)
            self._counts[layer] = 1
        else:
            self.mean_gap[layer] = mu * float(self.mean_gap[layer]) + (1.0 - mu) * float(
                mean_gap
            )
            self.mean_nongap[layer] = mu * float(self.mean_nongap[layer]) + (
                1.0 - mu
            ) * float(mean_nongap)
            self._counts[layer] = n + 1
        self.history.append(
            {
                "gap": dict(self.mean_gap),
                "nongap": dict(self.mean_nongap),
            }
        )

    def fallback_straddle_B(self, layer: str) -> float | None:
        """B = −(μ_gap+μ_nongap)/2 if layer has a reference; else None."""
        if layer not in self.mean_gap or layer not in self.mean_nongap:
            return None
        return -0.5 * (float(self.mean_gap[layer]) + float(self.mean_nongap[layer]))

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            "mean_gap": dict(self.mean_gap),
            "mean_nongap": dict(self.mean_nongap),
            "counts": {h: int(self._counts.get(h, 0)) for h in HIDDEN_LAYERS},
        }


@dataclass
class SelfCalibrateState:
    """Bundle of trackers owned by the orchestrator."""

    flags: SelfCalibrateFlags = field(default_factory=SelfCalibrateFlags)
    rates: LayerRateTracker | None = None
    prune: PruneCutoffTracker | None = None
    straddle: StraddleRefTracker | None = None

    @classmethod
    def from_config(
        cls,
        flags: SelfCalibrateFlags | Mapping[str, object] | None,
        *,
        mu_r_star: float = 0.999,
        mu_prune: float = 0.999,
        mu_straddle: float = 0.99,
        hybrid_period: int | None = None,
        prune_target_quantile: float = 0.1587,
    ) -> SelfCalibrateState:
        f = (
            flags
            if isinstance(flags, SelfCalibrateFlags)
            else SelfCalibrateFlags.from_mapping(flags)
        )
        state = cls(flags=f)
        if f.r_star:
            state.rates = LayerRateTracker(mu=mu_r_star, hybrid_period=hybrid_period)
        if f.prune_cutoff:
            state.prune = PruneCutoffTracker(
                mu=mu_prune,
                hybrid_period=hybrid_period,
                target_quantile=prune_target_quantile,
            )
        if f.straddle:
            state.straddle = StraddleRefTracker(mu=mu_straddle)
        return state

    def needs_warm_start(self) -> bool:
        if self.flags.r_star and (self.rates is None or not self.rates.warmed):
            return True
        if self.flags.prune_cutoff and (self.prune is None or not self.prune.warmed):
            return True
        # straddle may warm on first birth
        return False


def detect_rate_runaway(
    history: list[dict[str, float]],
    *,
    monotonic_epochs: int = 8,
    rel_drift: float = 0.30,
) -> dict[str, object]:
    """Check coupled-dynamics runaway on r* trajectory (per-layer)."""
    if len(history) < monotonic_epochs + 1:
        return {"runaway": False, "reason": "too_short", "layers": {}}
    layers_out: dict[str, object] = {}
    any_run = False
    for h in HIDDEN_LAYERS:
        series = [float(step[h]) for step in history if h in step]
        if len(series) < monotonic_epochs + 1:
            continue
        tail = series[-(monotonic_epochs + 1) :]
        diffs = [tail[i + 1] - tail[i] for i in range(len(tail) - 1)]
        all_up = all(d > 0 for d in diffs)
        all_down = all(d < 0 for d in diffs)
        start, end = series[0], series[-1]
        drift = abs(end - start) / max(abs(start), 1e-8)
        mono = all_up or all_down
        runaway = bool(mono and drift >= rel_drift)
        layers_out[h] = {
            "monotonic_tail": mono,
            "direction": "up" if all_up else ("down" if all_down else "mixed"),
            "rel_drift": drift,
            "runaway": runaway,
            "start": start,
            "end": end,
        }
        any_run = any_run or runaway
    return {"runaway": any_run, "layers": layers_out}


def detect_stat_oscillation(
    series: list[float],
    *,
    rel_step_change: float = 0.15,
    consecutive_steps: int = 3,
    last_n: int = 5,
    range_over_mean: float = 0.25,
) -> dict[str, object]:
    """Oscillation / thrash of a scalar running statistic."""
    if len(series) < 2:
        return {"oscillating": False}
    rel = []
    for i in range(1, len(series)):
        prev = abs(series[i - 1]) + 1e-8
        rel.append(abs(series[i] - series[i - 1]) / prev)
    run = 0
    max_run = 0
    for r in rel:
        if r >= rel_step_change:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    tail = series[-last_n:] if len(series) >= last_n else series
    mean = sum(tail) / len(tail)
    rng = max(tail) - min(tail)
    rom = rng / (abs(mean) + 1e-8)
    oscillating = max_run >= consecutive_steps or rom >= range_over_mean
    return {
        "oscillating": oscillating,
        "max_rel_run": max_run,
        "last_range_over_mean": rom,
        "mean_tail": mean,
    }
