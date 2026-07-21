"""Fixed vs metaplastic (BCM-style sliding) thresholds for Phase 1a.

Both variants share the same growth/contraction *meaning*; only how the
decision boundary is obtained differs.

Homeostatic targets are **per hidden layer** (r*_ℓ), not a single global scalar,
so layer-scale offsets (later layers run hotter) do not masquerade as
task-driven pressure — same category of issue Phase 0 fixed with per-layer
z-scoring before concatenated correlations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

import torch
from torch import Tensor

from neuronca.models.dynamic_mlp import HIDDEN_LAYERS


def normalize_target_rates(
    target_rate: float | Mapping[str, float] | None = None,
    *,
    target_rates: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Return ``{h1,h2,h3 → r*_ℓ}``.

    Accepts a single scalar (broadcast to all layers — discouraged for primary
    runs) or an explicit per-layer mapping. ``target_rates`` wins if both given.
    """
    if target_rates is not None:
        out = {h: float(target_rates[h]) for h in HIDDEN_LAYERS if h in target_rates}
        missing = [h for h in HIDDEN_LAYERS if h not in out]
        if missing:
            # Allow fc1/fc2/fc3 aliases
            aliases = {"fc1": "h1", "fc2": "h2", "fc3": "h3"}
            for key, val in target_rates.items():
                h = aliases.get(str(key), str(key))
                if h in HIDDEN_LAYERS:
                    out[h] = float(val)
            missing = [h for h in HIDDEN_LAYERS if h not in out]
        if missing:
            raise ValueError(
                f"target_rates missing layers {missing}; need all of {list(HIDDEN_LAYERS)}"
            )
        return {h: float(out[h]) for h in HIDDEN_LAYERS}

    if target_rate is None:
        raise ValueError("provide target_rate (scalar) or target_rates (per-layer)")
    if isinstance(target_rate, Mapping):
        return normalize_target_rates(target_rates=target_rate)
    r = float(target_rate)
    return {h: r for h in HIDDEN_LAYERS}


class ThresholdPolicy(ABC):
    """Produces growth/contraction pressure flags and edge threshold values."""

    name: str

    @abstractmethod
    def update(
        self,
        neuron_signals: Mapping[str, Tensor],
        active: Mapping[str, Tensor],
    ) -> None:
        """Update internal state from current per-hidden-layer signal EMAs.

        Args:
            neuron_signals: ``h1|h2|h3 → [H_max]`` activation magnitudes.
            active: ``h1|h2|h3 → [H_max]`` bool masks.
        """

    @abstractmethod
    def growth_pressure(self, layer: str) -> Tensor:
        """Bool tensor ``[H_max]`` — True where growth pressure is active."""

    @abstractmethod
    def contraction_pressure(self, layer: str) -> Tensor:
        """Bool tensor ``[H_max]`` — True where contraction pressure is active."""

    @abstractmethod
    def edge_threshold_for_pair(
        self,
        growth_hidden_layers: list[tuple[str, int]],
    ) -> float:
        """Scalar bar that ``incentive`` must clear (after eligibility).

        ``growth_hidden_layers`` lists ``(layer, slot)`` endpoints under growth
        pressure (hidden only). Fixed I/O never appear here.
        """

    def reset_slot(self, layer: str, slot: int) -> None:
        """Clear per-slot adaptive state on neuron birth (default: no-op)."""
        return None

    def target_for(self, layer: str) -> float:
        """Homeostatic target r*_ℓ for ``layer`` (if the policy has one)."""
        raise NotImplementedError


class FixedThreshold(ThresholdPolicy):
    """Variant A: shared τ; per-layer homeostatic targets r*_ℓ."""

    name = "fixed"

    def __init__(
        self,
        tau: float,
        target_rate: float | Mapping[str, float] | None = None,
        *,
        capacities: Mapping[str, int] | None = None,
        target_rates: Mapping[str, float] | None = None,
    ) -> None:
        if tau < 0:
            raise ValueError(f"tau must be non-negative, got {tau}")
        self.tau = float(tau)
        self.target_rates = normalize_target_rates(
            target_rate, target_rates=target_rates
        )
        # convenience: mean of layer targets (not used for pressure decisions)
        self.target_rate = sum(self.target_rates.values()) / len(self.target_rates)
        self._growth: dict[str, Tensor] = {}
        self._contraction: dict[str, Tensor] = {}
        self._capacities = dict(capacities or {h: 0 for h in HIDDEN_LAYERS})

    def target_for(self, layer: str) -> float:
        return self.target_rates[layer]

    def update(
        self,
        neuron_signals: Mapping[str, Tensor],
        active: Mapping[str, Tensor],
    ) -> None:
        for layer in HIDDEN_LAYERS:
            s = neuron_signals[layer]
            a = active[layer]
            r = self.target_rates[layer]
            e = s - r
            self._growth[layer] = a & (e > self.tau)
            self._contraction[layer] = a & (e < -self.tau)

    def growth_pressure(self, layer: str) -> Tensor:
        return self._growth[layer]

    def contraction_pressure(self, layer: str) -> Tensor:
        return self._contraction[layer]

    def edge_threshold_for_pair(
        self,
        growth_hidden_layers: list[tuple[str, int]],
    ) -> float:
        del growth_hidden_layers
        return self.tau


class SlidingThreshold(ThresholdPolicy):
    """Variant B: per-neuron θ as slow EMA of own activation (BCM-style).

    Edge bar uses ``max(θ_g)`` over growth-pressure **hidden** endpoints only —
    never a product of thresholds (plan §Resolved #3).

    θ is initialized / reset to the layer's r*_ℓ (probe-locked set-point), not a
    global scalar.
    """

    name = "sliding"

    def __init__(
        self,
        mu: float,
        target_rate: float | Mapping[str, float],
        capacities: Mapping[str, int],
        *,
        target_rates: Mapping[str, float] | None = None,
        edge_tau_floor: float = 0.0,
        pressure_margin: float = 0.0,
    ) -> None:
        if not 0.0 < mu < 1.0:
            raise ValueError(f"mu must be in (0,1), got {mu}")
        self.mu = float(mu)
        if target_rates is not None:
            self.target_rates = normalize_target_rates(target_rates=target_rates)
        elif isinstance(target_rate, Mapping):
            self.target_rates = normalize_target_rates(target_rates=target_rate)
        else:
            self.target_rates = normalize_target_rates(float(target_rate))
        self.target_rate = sum(self.target_rates.values()) / len(self.target_rates)
        self.edge_tau_floor = float(edge_tau_floor)
        self.pressure_margin = float(pressure_margin)
        self._theta: dict[str, Tensor] = {
            h: torch.full(
                (int(capacities[h]),),
                float(self.target_rates[h]),
                dtype=torch.float32,
            )
            for h in HIDDEN_LAYERS
        }
        self._growth: dict[str, Tensor] = {}
        self._contraction: dict[str, Tensor] = {}

    def target_for(self, layer: str) -> float:
        return self.target_rates[layer]

    def update(
        self,
        neuron_signals: Mapping[str, Tensor],
        active: Mapping[str, Tensor],
    ) -> None:
        for layer in HIDDEN_LAYERS:
            s = neuron_signals[layer].detach().float().cpu()
            a = active[layer].detach().cpu()
            theta = self._theta[layer]
            if theta.shape != s.shape:
                raise ValueError(
                    f"theta shape {tuple(theta.shape)} != signal {tuple(s.shape)} for {layer}"
                )
            theta[a] = self.mu * theta[a] + (1.0 - self.mu) * s[a]
            margin = self.pressure_margin
            self._growth[layer] = a & (s > theta + margin)
            self._contraction[layer] = a & (s < theta - margin)

    def growth_pressure(self, layer: str) -> Tensor:
        return self._growth[layer]

    def contraction_pressure(self, layer: str) -> Tensor:
        return self._contraction[layer]

    def edge_threshold_for_pair(
        self,
        growth_hidden_layers: list[tuple[str, int]],
    ) -> float:
        if not growth_hidden_layers:
            return self.edge_tau_floor
        vals = [
            float(self._theta[layer][slot].item())
            for layer, slot in growth_hidden_layers
        ]
        return max(max(vals), self.edge_tau_floor)

    def reset_slot(self, layer: str, slot: int) -> None:
        self._theta[layer][slot] = self.target_rates[layer]

    def theta(self, layer: str) -> Tensor:
        return self._theta[layer]


def build_threshold_policy(
    kind: str,
    *,
    tau: float,
    target_rate: float | Mapping[str, float] | None = None,
    target_rates: Mapping[str, float] | None = None,
    mu: float = 0.999,
    capacities: Mapping[str, int] | None = None,
    edge_tau_floor: float | None = None,
    pressure_margin: float = 0.0,
) -> ThresholdPolicy:
    """Factory: ``kind`` is ``fixed`` or ``sliding``."""
    key = kind.strip().lower()
    rates = normalize_target_rates(target_rate, target_rates=target_rates)
    if key == "fixed":
        return FixedThreshold(
            tau=tau, target_rate=rates, capacities=capacities, target_rates=rates
        )
    if key == "sliding":
        if capacities is None:
            raise ValueError("sliding threshold requires capacities")
        return SlidingThreshold(
            mu=mu,
            target_rate=rates,
            capacities=capacities,
            target_rates=rates,
            edge_tau_floor=float(tau if edge_tau_floor is None else edge_tau_floor),
            pressure_margin=pressure_margin,
        )
    raise ValueError(f"unknown threshold kind {kind!r}; expected 'fixed' or 'sliding'")
