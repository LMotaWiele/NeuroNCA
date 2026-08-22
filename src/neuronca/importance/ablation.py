"""GT-2: ablation sensitivity; ranking is ``max(0, L_ablate - L_base)``.

Edge mask is fixed per seed so lag series stay comparable. GT must not advance
training RNG (``docs/design/invariants.md``).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
from torch import Tensor

from neuronca.importance.base import (
    ImportanceResult,
    ProbeSource,
    cpu_float,
    iter_linear_layers,
    mean_probe_loss,
    resolve_device,
)

# Default hidden Linear names for MLPReasoningNet neuron ablation.
DEFAULT_HIDDEN_LAYERS: tuple[str, ...] = ("fc1", "fc2", "fc3")
DEFAULT_EDGE_SAMPLE: int = 512
DEFAULT_EDGE_MASK_SEED_OFFSET: int = 17
DEFAULT_N_BINS: int = 8


def layer_id_hash(layer_name: str) -> int:
    """Stable non-cryptographic id hash for layer-local RNG seeds."""
    digest = hashlib.sha256(layer_name.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def edge_mask_rng_seed(
    seed: int,
    layer_name: str,
    *,
    edge_mask_seed_offset: int = DEFAULT_EDGE_MASK_SEED_OFFSET,
) -> int:
    """RNG seed for a layer's edge mask: ``seed + layer_id_hash + offset``."""
    # Keep in signed 63-bit range for torch Generators.
    raw = int(seed) + layer_id_hash(layer_name) + int(edge_mask_seed_offset)
    return raw % (2**63 - 1)


def build_edge_ablation_mask(
    model: nn.Module,
    *,
    seed: int,
    sample_per_layer: int = DEFAULT_EDGE_SAMPLE,
    n_bins: int = DEFAULT_N_BINS,
    edge_mask_seed_offset: int = DEFAULT_EDGE_MASK_SEED_OFFSET,
    layer_names: Iterable[str] | None = None,
) -> dict[str, Tensor]:
    """Draw a fixed boolean edge mask per Linear layer, stratified by |W|.

    Sampling is deterministic given ``(seed, layer_name, offset)`` and the
    current weight magnitudes (bins). Call once per experiment seed and reuse
    across snapshots — do **not** redraw.

    Returns:
        ``layer → bool[n_out, n_in]`` with exactly ``min(sample, numel)`` True
        entries (or all True when the layer is smaller than ``sample_per_layer``).
    """
    if sample_per_layer < 0:
        raise ValueError(f"sample_per_layer must be >= 0; got {sample_per_layer}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1; got {n_bins}")

    layers = iter_linear_layers(model)
    if layer_names is not None:
        allow = set(layer_names)
        layers = [(n, m) for n, m in layers if n in allow]

    masks: dict[str, Tensor] = {}
    for name, mod in layers:
        w = mod.weight.detach()
        n_out, n_in = w.shape
        numel = n_out * n_in
        mask = torch.zeros(n_out, n_in, dtype=torch.bool)
        if sample_per_layer == 0 or numel == 0:
            masks[name] = mask
            continue

        k = min(int(sample_per_layer), numel)
        g = torch.Generator(device="cpu")
        g.manual_seed(
            edge_mask_rng_seed(
                seed, name, edge_mask_seed_offset=edge_mask_seed_offset
            )
        )

        abs_flat = w.abs().reshape(-1).cpu()
        if k == numel:
            mask.fill_(True)
            masks[name] = mask
            continue

        # Stratify by |W| rank bins; sample ~evenly from each occupied bin.
        # Equal values → bins may be uneven; we still cap total at k.
        # qcut-like: rank then bucket into n_bins contiguous rank groups.
        order = torch.argsort(abs_flat)
        bin_ids = torch.empty(numel, dtype=torch.long)
        edges = torch.linspace(0, numel, n_bins + 1).long()
        for b in range(n_bins):
            lo, hi = int(edges[b].item()), int(edges[b + 1].item())
            if hi > lo:
                bin_ids[order[lo:hi]] = b

        selected: list[int] = []
        # Per-bin target; distribute remainder to higher |W| bins first.
        base = k // n_bins
        rem = k - base * n_bins
        for b in range(n_bins - 1, -1, -1):
            members = (bin_ids == b).nonzero(as_tuple=False).view(-1)
            if members.numel() == 0:
                continue
            take = base + (1 if rem > 0 else 0)
            if rem > 0:
                rem -= 1
            take = min(take, int(members.numel()))
            if take <= 0:
                continue
            perm = torch.randperm(members.numel(), generator=g)[:take]
            selected.extend(members[perm].tolist())

        # If short (empty bins), fill from remaining indices.
        if len(selected) < k:
            chosen = set(selected)
            rest = [i for i in range(numel) if i not in chosen]
            need = k - len(selected)
            if rest:
                extra_perm = torch.randperm(len(rest), generator=g)[:need]
                selected.extend(rest[int(i)] for i in extra_perm.tolist())

        selected = selected[:k]
        flat_mask = torch.zeros(numel, dtype=torch.bool)
        if selected:
            flat_mask[torch.tensor(selected, dtype=torch.long)] = True
        masks[name] = flat_mask.view(n_out, n_in)

    return masks


def save_edge_ablation_mask(
    mask: dict[str, Tensor],
    path: str | Path,
) -> None:
    """Persist edge ablation mask dict to ``path`` (torch.save)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v.detach().cpu().bool() for k, v in mask.items()}
    torch.save(payload, path)


def load_edge_ablation_mask(path: str | Path) -> dict[str, Tensor]:
    """Load a previously saved edge ablation mask."""
    obj = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise TypeError(f"expected dict mask file; got {type(obj)!r}")
    return {k: v.bool() for k, v in obj.items()}


def default_edge_mask_path(meta_dir: str | Path, seed: int) -> Path:
    """Design path: ``meta/edge_ablation_mask_seed{k}.pt``."""
    return Path(meta_dir) / f"edge_ablation_mask_seed{int(seed)}.pt"


class AblationImportance:
    """Ablation sensitivity GT (secondary). Name: ``ablation_relu``."""

    name: str = "ablation_relu"

    def __init__(
        self,
        *,
        edge_ablation_sample: int = DEFAULT_EDGE_SAMPLE,
        neuron_ablation: str = "all_hidden",
        rank_transform: str = "relu",
        seed: int = 0,
        edge_mask_seed_offset: int = DEFAULT_EDGE_MASK_SEED_OFFSET,
        n_bins: int = DEFAULT_N_BINS,
        hidden_layers: tuple[str, ...] = DEFAULT_HIDDEN_LAYERS,
    ) -> None:
        """Args:
            edge_ablation_sample: Edges to ablate per layer (0 skips edge abl).
            neuron_ablation: ``"all_hidden"`` (default) or ``"none"``.
            rank_transform: ``"relu"`` → max(0, ΔL); ``"none"`` keeps signed
                in primary fields (signed always in extras).
            seed: Experiment seed for mask draw when mask not supplied.
            edge_mask_seed_offset: Added into per-layer RNG seed (config default 17).
            n_bins: |W| stratification bins for edge sampling.
            hidden_layers: Linear names whose post units are neuron-ablated.
        """
        if neuron_ablation not in ("all_hidden", "none"):
            raise ValueError(
                f"neuron_ablation must be 'all_hidden' or 'none'; got {neuron_ablation!r}"
            )
        if rank_transform not in ("relu", "none", "abs"):
            raise ValueError(
                f"rank_transform must be 'relu', 'none', or 'abs'; got {rank_transform!r}"
            )
        self.edge_ablation_sample = int(edge_ablation_sample)
        self.neuron_ablation = neuron_ablation
        self.rank_transform = rank_transform
        self.seed = int(seed)
        self.edge_mask_seed_offset = int(edge_mask_seed_offset)
        self.n_bins = int(n_bins)
        self.hidden_layers = tuple(hidden_layers)
        # Cached fixed mask: drawn once per instance so multi-snapshot calls do
        # not rebuild from current |W| (design: fixed per-seed mask).
        self._cached_edge_sample_mask: dict[str, Tensor] | None = None

    def clear_edge_mask_cache(self) -> None:
        """Drop the instance-cached edge mask (next compute redraws if needed)."""
        self._cached_edge_sample_mask = None

    def compute(
        self,
        model: nn.Module,
        probe_loader: ProbeSource,
        criterion: nn.Module,
        *,
        edge_sample_mask: dict[str, Tensor] | None = None,
    ) -> ImportanceResult:
        """Neuron + (optional) edge ablation on the probe set.

        ``probe_loader`` may be a DataLoader or a snapshot-scoped
        :class:`~neuronca.importance.probe_cache.ProbeCache` (preferred for
        production snapshots so the transform pipeline runs once).

        Does not call any local-signal update path. Restores weights/hooks
        after each unit ablation so the model is left unchanged.

        **Fixed edge mask (multi-snapshot):** When ``edge_sample_mask`` is
        omitted, the first call draws a stratified mask and **caches** it on
        this instance for subsequent ``compute`` calls (same edges even if
        weights change). Multi-snapshot runners may also pass a prebuilt mask
        explicitly (or load via :func:`load_edge_ablation_mask`) — that path
        always wins and is not overwritten into the cache unless no cache
        exists yet. Use :meth:`clear_edge_mask_cache` to force a redraw.
        """
        device = resolve_device(model)
        model.to(device)
        was_training = model.training
        model.eval()

        l_base = mean_probe_loss(model, probe_loader, criterion, device=device)

        neuron: dict[str, Tensor] = {}
        neuron_signed: dict[str, Tensor] = {}
        if self.neuron_ablation == "all_hidden":
            neuron, neuron_signed = self._ablate_neurons(
                model, probe_loader, criterion, l_base, device
            )

        edge: dict[str, Tensor] = {}
        edge_signed: dict[str, Tensor] = {}
        edge_index: dict[str, Tensor] | None = None

        if self.edge_ablation_sample > 0 or edge_sample_mask is not None:
            mask = self._resolve_edge_mask(model, edge_sample_mask)
            edge, edge_signed, edge_index = self._ablate_edges(
                model,
                probe_loader,
                criterion,
                l_base,
                device,
                mask,
            )

        if was_training:
            model.train()

        # Nested structure avoids layer-key collision between neuron [n_out]
        # and edge [n_out, n_in] under the design diagnostic name.
        ablation_signed: dict[str, dict[str, Tensor]] = {
            "neuron": neuron_signed,
            "edge": edge_signed,
        }
        extras: dict[str, Any] = {
            "ablation_signed": ablation_signed,
            # Convenience aliases (same objects as nested maps).
            "ablation_signed_neuron": neuron_signed,
            "ablation_signed_edge": edge_signed,
            "l_base": {"value": torch.tensor([l_base], dtype=torch.float32)},
        }

        return ImportanceResult(
            neuron=neuron,
            edge=edge,
            edge_index=edge_index,
            extras=extras,
        )

    def _resolve_edge_mask(
        self,
        model: nn.Module,
        edge_sample_mask: dict[str, Tensor] | None,
    ) -> dict[str, Tensor]:
        """Return caller mask, cached mask, or newly drawn (then cached) mask."""
        if edge_sample_mask is not None:
            # Explicit mask always used; also seed the cache if empty so later
            # bare compute() calls stay aligned with the first explicit pass.
            if self._cached_edge_sample_mask is None:
                self._cached_edge_sample_mask = {
                    k: v.detach().cpu().bool().clone()
                    for k, v in edge_sample_mask.items()
                }
            return edge_sample_mask

        if self._cached_edge_sample_mask is not None:
            return self._cached_edge_sample_mask

        drawn = build_edge_ablation_mask(
            model,
            seed=self.seed,
            sample_per_layer=self.edge_ablation_sample,
            n_bins=self.n_bins,
            edge_mask_seed_offset=self.edge_mask_seed_offset,
        )
        self._cached_edge_sample_mask = {
            k: v.detach().cpu().bool().clone() for k, v in drawn.items()
        }
        return self._cached_edge_sample_mask

    def _apply_rank(self, delta: Tensor) -> Tensor:
        if self.rank_transform == "relu":
            return delta.clamp(min=0.0)
        if self.rank_transform == "abs":
            return delta.abs()
        return delta

    def _ablate_neurons(
        self,
        model: nn.Module,
        probe_loader: ProbeSource,
        criterion: nn.Module,
        l_base: float,
        device: torch.device,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Zero each hidden post channel via Linear forward hook (pre-ReLU).

        Zeroing Linear output channel j forces post-ReLU activation j to 0,
        matching post-unit ablation for ReLU MLPs.
        """
        layers = {n: m for n, m in iter_linear_layers(model)}
        neuron: dict[str, Tensor] = {}
        signed: dict[str, Tensor] = {}

        for layer_name in self.hidden_layers:
            if layer_name not in layers:
                continue
            mod = layers[layer_name]
            n_out = mod.out_features
            deltas = torch.zeros(n_out, dtype=torch.float32)

            for j in range(n_out):
                handle = mod.register_forward_hook(_make_zero_channel_hook(j))
                try:
                    l_ab = mean_probe_loss(
                        model, probe_loader, criterion, device=device
                    )
                finally:
                    handle.remove()
                deltas[j] = float(l_ab - l_base)

            signed[layer_name] = deltas.clone()
            neuron[layer_name] = self._apply_rank(deltas)

        return neuron, signed

    def _ablate_edges(
        self,
        model: nn.Module,
        probe_loader: ProbeSource,
        criterion: nn.Module,
        l_base: float,
        device: torch.device,
        edge_sample_mask: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        """Ablate each masked edge by temporarily zeroing that weight entry."""
        layers = {n: m for n, m in iter_linear_layers(model)}
        edge: dict[str, Tensor] = {}
        signed: dict[str, Tensor] = {}
        edge_index: dict[str, Tensor] = {}

        for layer_name, mask in edge_sample_mask.items():
            if layer_name not in layers:
                continue
            mod = layers[layer_name]
            mask_b = mask.to(device=device).bool()
            if mask_b.shape != mod.weight.shape:
                raise ValueError(
                    f"edge mask shape {tuple(mask_b.shape)} != weight "
                    f"{tuple(mod.weight.shape)} for layer {layer_name}"
                )
            edge_index[layer_name] = mask_b.detach().cpu().bool().clone()

            scores = torch.zeros_like(mod.weight, dtype=torch.float32, device="cpu")
            signed_full = torch.zeros_like(scores)
            rows, cols = mask_b.nonzero(as_tuple=True)
            # Iterate sampled edges; restore weight after each.
            for r, c in zip(rows.tolist(), cols.tolist()):
                with torch.no_grad():
                    original = mod.weight[r, c].item()
                    mod.weight[r, c] = 0.0
                try:
                    l_ab = mean_probe_loss(
                        model, probe_loader, criterion, device=device
                    )
                finally:
                    with torch.no_grad():
                        mod.weight[r, c] = original
                d = float(l_ab - l_base)
                signed_full[r, c] = d
                scores[r, c] = float(self._apply_rank(torch.tensor(d)).item())

            edge[layer_name] = scores
            signed[layer_name] = signed_full

        return edge, signed, edge_index


def _make_zero_channel_hook(channel: int):
    """Forward hook that zeros output[:, channel] (Linear out / pre-ReLU)."""

    def hook(_module: nn.Module, _inp: tuple, output: Tensor) -> Tensor:
        # Clone to avoid in-place on views that break autograd elsewhere;
        # ablation path runs under no_grad via mean_probe_loss.
        out = output.clone()
        out[:, channel] = 0.0
        return out

    return hook
