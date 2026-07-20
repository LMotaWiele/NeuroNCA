"""Local importance signal candidates (read-only observations)."""

from neuronca.signals.activation import ActivationMagnitude
from neuronca.signals.base import (
    DEFAULT_DECAY,
    LAYER_ACT_KEYS,
    ConfigError,
    LocalSignal,
    ScoreBundle,
)
from neuronca.signals.eligibility import ThreeFactorScalarEligibility
from neuronca.signals.gevb_eligibility import (
    GEVBVectorEligibility,
    default_gevb_persist_path,
    gevb_rng_seed,
)
from neuronca.signals.hebbian_corr import HebbianResidualEnergyCorr

__all__ = [
    "DEFAULT_DECAY",
    "LAYER_ACT_KEYS",
    "ActivationMagnitude",
    "ConfigError",
    "GEVBVectorEligibility",
    "HebbianResidualEnergyCorr",
    "LocalSignal",
    "ScoreBundle",
    "ThreeFactorScalarEligibility",
    "build_signal",
    "default_gevb_persist_path",
    "gevb_rng_seed",
]


def build_signal(name: str, *, decay: float = DEFAULT_DECAY, **kwargs) -> LocalSignal:
    """Construct a local signal candidate by config name."""
    if name == "activation":
        return ActivationMagnitude(decay=decay)
    if name == "hebbian_corr":
        return HebbianResidualEnergyCorr(decay=decay)
    if name == "eligibility_residual_energy":
        return ThreeFactorScalarEligibility(decay=decay)
    if name == "eligibility_gevb_rf":
        return GEVBVectorEligibility(decay=decay, **kwargs)
    raise ValueError(
        f"unknown signal candidate {name!r}; expected one of "
        f"activation, hebbian_corr, eligibility_residual_energy, eligibility_gevb_rf"
    )
