"""Full-state structure snapshot export for Phase 1a visualization.

Opt-in observability only: dumps per-neuron / per-edge state at chosen epochs.
Does not alter training, orchestrator behavior, or any pass/fail criterion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from neuronca.eval.schedule import stationary_snapshot_tags
from neuronca.models.dynamic_mlp import (
    HIDDEN_LAYERS,
    LINEAR_LAYERS,
    DynamicMLPReasoningNet,
    _LINEAR_ENDPOINTS,
)
from neuronca.plasticity.masks import (
    BIRTH_KIND_NAMES,
    ORIGIN_EDGE_NAMES,
    ORIGIN_NEURON_NAMES,
)

# Signal score keys are Linear names; map to hidden layer names.
_HIDDEN_TO_FC = {"h1": "fc1", "h2": "fc2", "h3": "fc3"}


def resolve_snapshot_epochs(
    total_epochs: int,
    cadence: str | Sequence[int] | Mapping[str, Any] | bool | None,
) -> dict[int, str]:
    """Map 0-based epoch index → tag for snapshot points.

    Cadence options:
      - True / ``"every_epoch"`` / ``None`` → every epoch, tags ``epoch_0``…
      - ``"early_mid_late"`` → schedule tags at end of early/mid/late epochs
        (1-based schedule epochs converted to 0-based end-of-epoch indices)
      - list/tuple of 0-based epochs
      - mapping with ``epochs`` list and optional ``tags``
    """
    if total_epochs < 1:
        return {}

    if cadence is None or cadence is True or cadence == "every_epoch":
        return {e: f"epoch_{e}" for e in range(total_epochs)}

    if cadence is False:
        return {}

    if isinstance(cadence, str):
        key = cadence.strip().lower()
        if key in ("every", "every_epoch", "all", "epoch"):
            return {e: f"epoch_{e}" for e in range(total_epochs)}
        if key in ("early_mid_late", "early/mid/late", "stationary"):
            out: dict[int, str] = {}
            for spec in stationary_snapshot_tags(epochs=total_epochs):
                # Schedule uses 1-based end-of-epoch numbering.
                ep1 = int(spec.epoch) if spec.epoch is not None else 1
                ep0 = max(0, min(total_epochs - 1, ep1 - 1))
                out[ep0] = spec.tag
            return out
        raise ValueError(f"unknown structure_snapshot_cadence: {cadence!r}")

    if isinstance(cadence, Mapping):
        epochs = cadence.get("epochs")
        tags = cadence.get("tags")
        if epochs is None and "cadence" in cadence:
            return resolve_snapshot_epochs(total_epochs, cadence["cadence"])
        if epochs is None:
            raise ValueError("structure_snapshot_cadence mapping needs 'epochs'")
        epochs_list = [int(e) for e in epochs]
        if tags is None:
            return {
                e: f"epoch_{e}"
                for e in epochs_list
                if 0 <= e < total_epochs
            }
        tag_list = list(tags)
        out = {}
        for i, e in enumerate(epochs_list):
            if 0 <= int(e) < total_epochs:
                out[int(e)] = str(tag_list[i]) if i < len(tag_list) else f"epoch_{e}"
        return out

    if isinstance(cadence, (list, tuple)):
        return {
            int(e): f"epoch_{int(e)}"
            for e in cadence
            if 0 <= int(e) < total_epochs
        }

    raise TypeError(
        f"structure_snapshot_cadence must be str/list/dict/bool, got {type(cadence)}"
    )


def _neuron_activation_ema(
    signal: Any | None,
    layer: str,
    slot: int,
) -> float | None:
    """Read ActivationMagnitude neuron EMA for a hidden slot, if available."""
    if signal is None:
        return None
    scores = None
    if hasattr(signal, "neuron_scores"):
        try:
            scores = signal.neuron_scores()
        except Exception:
            scores = None
    if not isinstance(scores, Mapping):
        return None
    fc = _HIDDEN_TO_FC.get(layer)
    if fc is None:
        return None
    # Scores may be keyed by fc* or h*.
    buf = scores.get(fc)
    if buf is None:
        buf = scores.get(layer)
    if buf is None:
        return None
    t = buf.detach().cpu().float().reshape(-1)
    if slot < 0 or slot >= int(t.numel()):
        return None
    return float(t[slot].item())


def _origin_name_from_codes(kind_code: int, origin_code: int) -> str:
    origin_name = BIRTH_KIND_NAMES.get(
        kind_code, ORIGIN_NEURON_NAMES.get(origin_code, "unknown")
    )
    if kind_code == 0 and origin_code == 1:
        origin_name = "init"
    return origin_name


def capture_structure_snapshot(
    net: DynamicMLPReasoningNet,
    *,
    epoch: int,
    tag: str,
    plasticity_step: int | None = None,
    signal: Any | None = None,
    specialization: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
    pruned_this_step: Sequence[tuple[str, int]] | None = None,
    pruned_records: Sequence[Mapping[str, Any]] | None = None,
    grown_this_step: Sequence[tuple[str, int]] | None = None,
) -> dict[str, Any]:
    """Direct state dump of active neurons + edges (not a delta reconstruction)."""
    step = (
        int(plasticity_step)
        if plasticity_step is not None
        else int(net.cooldown.plasticity_step)
    )
    pruned_set = set(pruned_this_step or ())
    grown_set = set(grown_this_step or ())
    layers_out: dict[str, Any] = {}

    for h in HIDDEN_LAYERS:
        active = net.active_mask(h).cpu()
        slots = active.nonzero(as_tuple=False).view(-1).tolist()
        coords = net.coords(h).cpu().float()
        origins = net.neuron_origin(h).cpu()
        kinds = net.birth_kind(h).cpu()
        parents = net.parent_slot(h).cpu()
        births = net.birth_step(h).cpu()
        neurons: list[dict[str, Any]] = []
        for s in slots:
            s = int(s)
            kind_code = int(kinds[s].item())
            origin_code = int(origins[s].item())
            parent = int(parents[s].item())
            bstep = int(births[s].item())
            age = (step - bstep) if bstep >= 0 else -1
            origin_name = _origin_name_from_codes(kind_code, origin_code)
            row: dict[str, Any] = {
                "id": s,
                "x": float(coords[s, 0].item()),
                "y": float(coords[s, 1].item()),
                "origin": origin_name,
                "neuron_origin": ORIGIN_NEURON_NAMES.get(origin_code, "unknown"),
                "birth_kind": BIRTH_KIND_NAMES.get(kind_code, "none"),
                "age": int(age),
                "birth_step": bstep,
                "parent_slot": parent if parent >= 0 else None,
                "activation_ema": _neuron_activation_ema(signal, h, s),
                "grown_this_step": (h, s) in grown_set,
            }
            if specialization is not None and (h, s) in specialization:
                spec = specialization[(h, s)]
                for key in (
                    "gap_over_whole",
                    "on_off",
                    "gap_whole",
                    "mean_post_gap",
                    "mean_post_whole",
                ):
                    if key in spec and spec[key] is not None:
                        # Normalize alias gap_whole → gap_over_whole in dump.
                        out_key = "gap_over_whole" if key == "gap_whole" else key
                        row[out_key] = float(spec[key])
            neurons.append(row)
        layers_out[h] = {"neurons": neurons}

    edges_out: dict[str, list[dict[str, Any]]] = {}
    for lin in LINEAR_LAYERS:
        pre, post = _LINEAR_ENDPOINTS[lin]
        em = net.edge_mask(lin).cpu()
        eo = net.edge_origin(lin).cpu()
        pre_a = net.population_active(pre).cpu()
        post_a = net.population_active(post).cpu()
        used = em & post_a.unsqueeze(1) & pre_a.unsqueeze(0)
        if not bool(used.any()):
            edges_out[lin] = []
            continue
        oi, ii = used.nonzero(as_tuple=True)
        edge_list: list[dict[str, Any]] = []
        for o, i in zip(oi.tolist(), ii.tolist()):
            code = int(eo[o, i].item())
            edge_list.append(
                {
                    "i": int(i),  # pre index
                    "j": int(o),  # post index
                    "pre_pop": pre,
                    "post_pop": post,
                    "origin": ORIGIN_EDGE_NAMES.get(code, f"origin_{code}"),
                }
            )
        edges_out[lin] = edge_list

    # Pruned units are no longer active; keep last-living coords for death markers.
    pruned_neurons: list[dict[str, Any]] = []
    if pruned_records:
        for rec in pruned_records:
            layer = str(rec.get("layer"))
            slot = int(rec["slot"])
            kind_code = int(rec.get("birth_kind", 0) or 0)
            origin_code = int(rec.get("neuron_origin", 0) or 0)
            parent = rec.get("parent_slot")
            parent_i = int(parent) if parent is not None and int(parent) >= 0 else None
            bstep = rec.get("birth_step")
            age = (
                (step - int(bstep))
                if bstep is not None and int(bstep) >= 0
                else -1
            )
            pruned_neurons.append(
                {
                    "layer": layer,
                    "id": slot,
                    "x": rec.get("x"),
                    "y": rec.get("y"),
                    "origin": _origin_name_from_codes(kind_code, origin_code),
                    "neuron_origin": ORIGIN_NEURON_NAMES.get(origin_code, "unknown"),
                    "birth_kind": BIRTH_KIND_NAMES.get(kind_code, "none"),
                    "age": int(age),
                    "birth_step": bstep,
                    "parent_slot": parent_i,
                    "pruned_this_step": True,
                }
            )
            pruned_set.add((layer, slot))
    elif pruned_set:
        # Fallback without coords (legacy callers) — still list keys.
        pruned_neurons = [
            {"layer": h, "id": int(s), "x": None, "y": None, "pruned_this_step": True}
            for h, s in sorted(pruned_set)
        ]

    return {
        "schema": "neuronca.structure_snapshot.v1",
        "epoch": int(epoch),
        "tag": str(tag),
        "plasticity_step": step,
        "hidden_counts": net.hidden_counts(),
        "edge_counts": net.edge_counts(by_origin=True),
        "layers": layers_out,
        "edges": edges_out,
        # Pruned units with last-living coords (for open-circle death marks).
        "pruned_this_step": [
            {"layer": h, "slot": int(s)} for h, s in sorted(pruned_set)
        ],
        "pruned_neurons": pruned_neurons,
        "grown_this_step": [
            {"layer": h, "slot": int(s)} for h, s in sorted(grown_set)
        ],
    }


def write_structure_snapshot(
    out_dir: str | Path,
    snapshot: Mapping[str, Any],
    *,
    filename: str | None = None,
) -> Path:
    """Write one snapshot JSON under ``out_dir/structure_snapshots/``."""
    root = Path(out_dir)
    snap_dir = root / "structure_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    epoch = int(snapshot.get("epoch", 0))
    tag = str(snapshot.get("tag", f"epoch_{epoch}"))
    if filename is None:
        # Zero-pad epoch for stable lexicographic order.
        filename = f"ep{epoch:04d}_{tag}.json"
    path = snap_dir / filename
    path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
    # Maintain a lightweight index for the visualizer.
    index_path = snap_dir / "index.json"
    index: dict[str, Any]
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            index = {"snapshots": []}
    else:
        index = {"snapshots": []}
    entries: list[dict[str, Any]] = list(index.get("snapshots") or [])
    entries = [e for e in entries if e.get("file") != filename]
    entries.append(
        {
            "epoch": epoch,
            "tag": tag,
            "file": filename,
            "hidden_counts": snapshot.get("hidden_counts"),
            "n_edges": {
                k: (v.get("total") if isinstance(v, dict) else v)
                for k, v in (snapshot.get("edge_counts") or {}).items()
            },
        }
    )
    entries.sort(key=lambda e: (int(e.get("epoch", 0)), str(e.get("tag", ""))))
    index["snapshots"] = entries
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return path


def load_structure_snapshot(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def list_structure_snapshots(results_root: str | Path) -> list[dict[str, Any]]:
    """Return index entries for snapshots under a seed/results root."""
    root = Path(results_root)
    candidates = [
        root / "structure_snapshots" / "index.json",
        root / "index.json",
    ]
    for p in candidates:
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            snaps = data.get("snapshots") or []
            # Resolve file paths relative to the snapshots directory.
            snap_dir = p.parent
            out = []
            for e in snaps:
                entry = dict(e)
                entry["path"] = str(snap_dir / e["file"])
                out.append(entry)
            return out
    # Fallback: scan directory for ep*.json
    snap_dir = root / "structure_snapshots"
    if not snap_dir.is_dir():
        snap_dir = root
    files = sorted(snap_dir.glob("ep*.json"))
    out = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        out.append(
            {
                "epoch": data.get("epoch"),
                "tag": data.get("tag"),
                "file": f.name,
                "path": str(f),
            }
        )
    return out


@dataclass
class StructureSnapshotRecorder:
    """Opt-in helper: write snapshots at configured epochs during training."""

    out_dir: Path
    epoch_tags: dict[int, str] = field(default_factory=dict)
    enabled: bool = True
    specialization: dict[tuple[str, int], dict[str, Any]] = field(default_factory=dict)
    written: list[Path] = field(default_factory=list)

    @classmethod
    def from_config(
        cls,
        out_dir: str | Path,
        *,
        total_epochs: int,
        record: bool = False,
        cadence: str | Sequence[int] | Mapping[str, Any] | bool | None = "every_epoch",
    ) -> StructureSnapshotRecorder | None:
        if not record:
            return None
        tags = resolve_snapshot_epochs(total_epochs, cadence)
        if not tags:
            return None
        return cls(out_dir=Path(out_dir), epoch_tags=tags, enabled=True)

    def set_specialization(
        self,
        layer: str,
        slot: int,
        metrics: Mapping[str, Any],
    ) -> None:
        self.specialization[(layer, int(slot))] = dict(metrics)

    def maybe_record(
        self,
        epoch: int,
        net: DynamicMLPReasoningNet,
        *,
        signal: Any | None = None,
        structure_log: Any | None = None,
    ) -> Path | None:
        if not self.enabled or epoch not in self.epoch_tags:
            return None
        tag = self.epoch_tags[epoch]
        pruned: list[tuple[str, int]] = []
        pruned_records: list[dict[str, Any]] = []
        grown: list[tuple[str, int]] = []
        if structure_log is not None:
            pruned = list(getattr(structure_log, "neuron_prune_keys", []) or [])
            pruned_records = list(
                getattr(structure_log, "neuron_prune_records", []) or []
            )
            grown = list(getattr(structure_log, "neuron_grow_keys", []) or [])
        snap = capture_structure_snapshot(
            net,
            epoch=epoch,
            tag=tag,
            signal=signal,
            specialization=self.specialization,
            pruned_this_step=pruned,
            pruned_records=pruned_records,
            grown_this_step=grown,
        )
        path = write_structure_snapshot(self.out_dir, snap)
        self.written.append(path)
        return path


__all__ = [
    "StructureSnapshotRecorder",
    "capture_structure_snapshot",
    "write_structure_snapshot",
    "load_structure_snapshot",
    "list_structure_snapshots",
    "resolve_snapshot_epochs",
]
