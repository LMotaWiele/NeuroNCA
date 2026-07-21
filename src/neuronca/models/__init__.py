"""Reasoning networks (fixed Phase 0 MLP and Phase 1a dynamic masked MLP)."""

from neuronca.models.dynamic_mlp import DynamicMLPReasoningNet
from neuronca.models.mlp import MLPReasoningNet

__all__ = ["MLPReasoningNet", "DynamicMLPReasoningNet"]