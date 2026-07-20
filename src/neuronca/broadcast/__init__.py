"""Global broadcast residual / error encoding."""

from neuronca.broadcast.error import (
    BroadcastErrorEncoder,
    BroadcastPacket,
    batch_mean_residual_as_eligibility_modulator,
)

__all__ = [
    "BroadcastErrorEncoder",
    "BroadcastPacket",
    "batch_mean_residual_as_eligibility_modulator",
]
