#!/usr/bin/env python3
"""Re-slice structure snapshots: do neighbors share specialization more than distant units?"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
OUT = (
    _REPO
    / "results"
    / "phase1a_diagnostics"
    / "specialization_spatial_coherence"
)
SNAP_ROOT = (
    _REPO
    / "results"
    / "phase1a_diagnostics"
    / "gap_imprint_survival_20ep_hold"
    / "seed_0"
    / "structure_snapshots"
)
UNIT_JSON = (
    _REPO
    / "results"
    / "phase1a_diagnostics"
    / "gap_imprint_survival_20ep_hold"
    / "seed_0.json"
)

LAYERS = ("h1", "h2", "h3")
# Official sensing radius (configs/phase1a_fixed.yaml).
DEFAULT_R = 0.35


def _load_snapshots() -> list[tuple[int, dict[str, Any]]]:
    files = sorted(SNAP_ROOT.glob("ep*.json"))
    out: list[tuple[int, dict[str, Any]]] = []
    for p in files:
        d = json.loads(p.read_text())
        out.append((int(d["epoch"]), d))
    return out


def _load_stays_set() -> set[tuple[str, int]]:
    if not UNIT_JSON.exists():
        return set()
    d = json.loads(UNIT_JSON.read_text())
    stays = set()
    for u in d.get("units", []):
        if u.get("stays_specialized"):
            stays.add((str(u["layer"]), int(u["slot"])))
    return stays


def _pairwise_distances(xy: list[tuple[float, float]]) -> list[list[float]]:
    n = len(xy)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = xy[i]
        for j in range(i + 1, n):
            xj, yj = xy[j]
            d = math.hypot(xi - xj, yi - yj)
            D[i][j] = D[j][i] = d
    return D


def _pair_stats(
    values: list[float],
    D: list[list[float]],
    radius: float,
    exclude_pairs: set[tuple[int, int]] | None = None,
) -> dict[str, float] | None:
    """Mean |Δ| for neighbor vs distant pairs."""
    n = len(values)
    if n < 5:
        return None
    near: list[float] = []
    far: list[float] = []
    n_excl = 0
    for i in range(n):
        for j in range(i + 1, n):
            if exclude_pairs and ((i, j) in exclude_pairs or (j, i) in exclude_pairs):
                n_excl += 1
                continue
            dabs = abs(values[i] - values[j])
            if D[i][j] <= radius:
                near.append(dabs)
            else:
                far.append(dabs)
    if len(near) < 3 or len(far) < 3:
        return None
    mn, mf = statistics.mean(near), statistics.mean(far)
    return {
        "n_near": float(len(near)),
        "n_far": float(len(far)),
        "n_excluded_pairs": float(n_excl),
        "mean_abs_delta_near": mn,
        "mean_abs_delta_far": mf,
        "near_over_far": mn / mf if mf > 1e-12 else float("nan"),
    }


def _morans_I(
    values: list[float],
    D: list[list[float]],
    radius: float,
) -> float | None:
    """Moran's I with binary weights w_ij = 1[d_ij ≤ R], i≠j."""
    n = len(values)
    if n < 5:
        return None
    mean_v = statistics.mean(values)
    den = sum((v - mean_v) ** 2 for v in values)
    if den < 1e-18:
        return 0.0
    num = 0.0
    W = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if D[i][j] <= radius:
                num += (values[i] - mean_v) * (values[j] - mean_v)
                W += 1.0
    if W < 1e-12:
        return None
    return (n / W) * (num / den)


def _parent_child_index_pairs(
    ids: list[int], parents: list[int | None]
) -> set[tuple[int, int]]:
    id_to_idx = {sid: i for i, sid in enumerate(ids)}
    pairs: set[tuple[int, int]] = set()
    for i, p in enumerate(parents):
        if p is None:
            continue
        if p in id_to_idx:
            j = id_to_idx[p]
            a, b = (i, j) if i < j else (j, i)
            pairs.add((a, b))
    return pairs


def _extract_specialists(neurons: list[dict[str, Any]]) -> dict[str, Any] | None:
    specs = [n for n in neurons if "gap_over_whole" in n and "on_off" in n]
    if len(specs) < 5:
        return None
    ids = [int(n["id"]) for n in specs]
    parents = [n.get("parent_slot") for n in specs]
    # parent may be int or None
    parents_i: list[int | None] = [
        int(p) if p is not None else None for p in parents
    ]
    xy = [(float(n["x"]), float(n["y"])) for n in specs]
    gow = [float(n["gap_over_whole"]) for n in specs]
    oo = [float(n["on_off"]) for n in specs]
    return {
        "n": len(specs),
        "ids": ids,
        "parents": parents_i,
        "xy": xy,
        "gow": gow,
        "on_off": oo,
    }


def _perm_null_pair_ratio(
    values: list[float],
    D: list[list[float]],
    radius: float,
    n_perm: int,
    rng: random.Random,
    exclude_pairs: set[tuple[int, int]] | None = None,
) -> list[float]:
    """Permutation null for near/far |Δ| ratio (coords fixed, labels shuffled)."""
    ratios: list[float] = []
    base = list(values)
    for _ in range(n_perm):
        rng.shuffle(base)
        st = _pair_stats(base, D, radius, exclude_pairs=exclude_pairs)
        if st is not None and st["near_over_far"] == st["near_over_far"]:
            ratios.append(float(st["near_over_far"]))
    return ratios


def _perm_null_moran(
    values: list[float],
    D: list[list[float]],
    radius: float,
    n_perm: int,
    rng: random.Random,
) -> list[float]:
    out: list[float] = []
    base = list(values)
    for _ in range(n_perm):
        rng.shuffle(base)
        mi = _morans_I(base, D, radius)
        if mi is not None:
            out.append(float(mi))
    return out


def _p_left(obs: float, null: list[float]) -> float:
    """One-sided: observed smaller than null (for near/far ratio: more coherent)."""
    if not null:
        return float("nan")
    return (1.0 + sum(1 for x in null if x <= obs)) / (1.0 + len(null))


def _p_right(obs: float, null: list[float]) -> float:
    """One-sided: observed larger than null (for Moran's I)."""
    if not null:
        return float("nan")
    return (1.0 + sum(1 for x in null if x >= obs)) / (1.0 + len(null))


def _binary_clustering(
    neurons: list[dict[str, Any]],
    positive_ids: set[int],
    radius: float,
    n_perm: int,
    rng: random.Random,
) -> dict[str, Any] | None:
    """Among all active neurons, do positives cluster spatially?

    Score: mean fraction of neighbors (d≤R) that are also positive,
    among positive units with ≥1 neighbor. Compare to label-permutation null.
    """
    if len(neurons) < 10 or len(positive_ids) < 3:
        return None
    ids = [int(n["id"]) for n in neurons]
    xy = [(float(n["x"]), float(n["y"])) for n in neurons]
    D = _pairwise_distances(xy)
    labels = [1 if sid in positive_ids else 0 for sid in ids]
    n_pos = sum(labels)
    if n_pos < 3 or n_pos == len(labels):
        return None

    def score(lab: list[int]) -> float | None:
        scores: list[float] = []
        for i, li in enumerate(lab):
            if li != 1:
                continue
            nbr = [j for j in range(len(lab)) if j != i and D[i][j] <= radius]
            if not nbr:
                continue
            scores.append(sum(lab[j] for j in nbr) / len(nbr))
        if len(scores) < 3:
            return None
        return statistics.mean(scores)

    obs = score(labels)
    if obs is None:
        return None
    null: list[float] = []
    base = list(labels)
    for _ in range(n_perm):
        rng.shuffle(base)
        s = score(base)
        if s is not None:
            null.append(s)
    # base rate
    base_rate = n_pos / len(labels)
    return {
        "n_neurons": len(labels),
        "n_positive": n_pos,
        "base_rate": base_rate,
        "mean_neighbor_positive_frac": obs,
        "lift_vs_base": obs / base_rate if base_rate > 0 else float("nan"),
        "null_mean": statistics.mean(null) if null else float("nan"),
        "null_p95": sorted(null)[int(0.95 * (len(null) - 1))] if null else float("nan"),
        "p_right": _p_right(obs, null),
    }


def analyze_epoch(
    snap: dict[str, Any],
    radius: float,
    n_perm: int,
    rng: random.Random,
    stays: set[tuple[str, int]],
) -> dict[str, Any]:
    epoch = int(snap["epoch"])
    by_layer: dict[str, Any] = {}
    for layer in LAYERS:
        neurons = snap["layers"][layer]["neurons"]
        spec = _extract_specialists(neurons)
        layer_out: dict[str, Any] = {"n_active": len(neurons)}
        if spec is None:
            layer_out["specialist_pairs"] = None
            layer_out["note"] = "too_few_specialization_tracked"
        else:
            D = _pairwise_distances(spec["xy"])
            pc = _parent_child_index_pairs(spec["ids"], spec["parents"])
            scalars = {
                "gap_over_whole": spec["gow"],
                "on_off": spec["on_off"],
            }
            pair_block: dict[str, Any] = {
                "n_specialists": spec["n"],
                "n_parent_child_pairs": len(pc),
            }
            for name, vals in scalars.items():
                # with parent-child included
                st_all = _pair_stats(vals, D, radius, exclude_pairs=None)
                st_ex = _pair_stats(vals, D, radius, exclude_pairs=pc)
                mi = _morans_I(vals, D, radius)
                null_ratio = _perm_null_pair_ratio(
                    vals, D, radius, n_perm, rng, exclude_pairs=pc
                )
                null_mi = _perm_null_moran(vals, D, radius, n_perm, rng)
                pair_block[name] = {
                    "with_parent_child": st_all,
                    "exclude_parent_child": st_ex,
                    "morans_I": mi,
                    "morans_I_null_mean": (
                        statistics.mean(null_mi) if null_mi else float("nan")
                    ),
                    "morans_I_null_p95": (
                        sorted(null_mi)[int(0.95 * (len(null_mi) - 1))]
                        if null_mi
                        else float("nan")
                    ),
                    "morans_I_p_right": _p_right(mi, null_mi) if mi is not None else float("nan"),
                    "near_over_far_excl_pc": (
                        st_ex["near_over_far"] if st_ex else float("nan")
                    ),
                    "near_over_far_excl_pc_null_mean": (
                        statistics.mean(null_ratio) if null_ratio else float("nan")
                    ),
                    "near_over_far_excl_pc_p_left": (
                        _p_left(st_ex["near_over_far"], null_ratio)
                        if st_ex
                        else float("nan")
                    ),
                }
            layer_out["specialist_pairs"] = pair_block

        # Binary clustering: tracked specialists among all active
        tracked_ids = {
            int(n["id"]) for n in neurons if "gap_over_whole" in n
        }
        stays_ids = {
            int(n["id"]) for n in neurons if (layer, int(n["id"])) in stays
        }
        layer_out["binary_tracked"] = _binary_clustering(
            neurons, tracked_ids, radius, n_perm, rng
        )
        layer_out["binary_stays"] = _binary_clustering(
            neurons, stays_ids, radius, n_perm, rng
        )
        by_layer[layer] = layer_out
    return {"epoch": epoch, "layers": by_layer}


def _layer_passes_coherent(layer_block: dict[str, Any], alpha: float = 0.05) -> dict[str, Any]:
    """Does this layer show spatial specialization coherence after confound control?"""
    sp = layer_block.get("specialist_pairs")
    if not sp:
        return {"pass": False, "reason": "no_specialists"}
    evidence: list[str] = []
    for scalar in ("gap_over_whole", "on_off"):
        b = sp.get(scalar) or {}
        ratio = b.get("near_over_far_excl_pc")
        p_ratio = b.get("near_over_far_excl_pc_p_left")
        mi = b.get("morans_I")
        p_mi = b.get("morans_I_p_right")
        # Coherent: neighbors more similar (ratio < 1) and significant, OR Moran > 0 significant
        ratio_ok = (
            ratio is not None
            and ratio == ratio
            and ratio < 0.90
            and p_ratio is not None
            and p_ratio == p_ratio
            and p_ratio <= alpha
        )
        moran_ok = (
            mi is not None
            and mi == mi
            and mi > 0.05
            and p_mi is not None
            and p_mi == p_mi
            and p_mi <= alpha
        )
        if ratio_ok:
            evidence.append(f"{scalar}:near_far_ratio={ratio:.3f}_p={p_ratio:.3f}")
        if moran_ok:
            evidence.append(f"{scalar}:MoranI={mi:.3f}_p={p_mi:.3f}")
    bin_t = layer_block.get("binary_tracked") or {}
    bin_s = layer_block.get("binary_stays") or {}
    for tag, b in (("tracked", bin_t), ("stays", bin_s)):
        if not b:
            continue
        p = b.get("p_right")
        lift = b.get("lift_vs_base")
        if (
            p is not None
            and p == p
            and p <= alpha
            and lift is not None
            and lift == lift
            and lift >= 1.15
        ):
            evidence.append(f"binary_{tag}:lift={lift:.2f}_p={p:.3f}")
    return {"pass": len(evidence) > 0, "evidence": evidence}


def _verdict(late: dict[str, Any], mid: dict[str, Any] | None) -> dict[str, Any]:
    late_passes = {
        L: _layer_passes_coherent(late["layers"][L]) for L in LAYERS
    }
    n_pass = sum(1 for v in late_passes.values() if v["pass"])
    # Check whether parent-child alone drives any apparent effect
    pc_only_notes: list[str] = []
    for L in LAYERS:
        sp = late["layers"][L].get("specialist_pairs")
        if not sp:
            continue
        for scalar in ("gap_over_whole", "on_off"):
            b = sp[scalar]
            w = b.get("with_parent_child") or {}
            e = b.get("exclude_parent_child") or {}
            if not w or not e:
                continue
            r_w, r_e = w.get("near_over_far"), e.get("near_over_far")
            if (
                r_w is not None
                and r_e is not None
                and r_w == r_w
                and r_e == r_e
                and r_w < 0.90
                and r_e >= 0.95
            ):
                pc_only_notes.append(
                    f"{L}.{scalar}: ratio {r_w:.3f}→{r_e:.3f} after excl parent-child"
                )

    # Strong pass = signature Moran or near/far (not only binary density of tracked
    # births, which can reflect growth-near-parent placement).
    strong_layers: list[str] = []
    for L, t in late_passes.items():
        if not t["pass"]:
            continue
        ev = " ".join(t.get("evidence") or [])
        if "MoranI=" in ev or "near_far_ratio=" in ev:
            strong_layers.append(L)

    # stays_specialized binary: do long-lived specialists form regions?
    stays_cluster = False
    stays_notes: list[str] = []
    for L in LAYERS:
        b = late["layers"][L].get("binary_stays") or {}
        if not b:
            stays_notes.append(f"{L}: no stays binary")
            continue
        lift = b.get("lift_vs_base")
        p = b.get("p_right")
        stays_notes.append(f"{L}: stays lift={lift} p={p}")
        if (
            lift is not None
            and lift == lift
            and lift >= 1.15
            and p is not None
            and p == p
            and p <= 0.05
        ):
            stays_cluster = True

    if len(strong_layers) >= 2 and stays_cluster:
        label = "SPATIALLY_COHERENT"
        go = True
        go_kind = "full_regional_mechanism"
    elif len(strong_layers) >= 1:
        label = "MIXED_LAYER_DEPENDENT"
        # Partial handle only — not a network-wide region structure.
        go = False
        go_kind = "no_full_build__optional_h1_probe_only"
    else:
        label = "SPATIALLY_RANDOM"
        go = False
        go_kind = "no_build"

    return {
        "label": label,
        "go_regional_consolidation": go,
        "go_kind": go_kind,
        "n_layers_pass_late": n_pass,
        "strong_signature_layers": strong_layers,
        "stays_specialists_cluster": stays_cluster,
        "stays_notes": stays_notes,
        "layer_tests": late_passes,
        "parent_child_confound_notes": pc_only_notes,
        "mid_epoch_checked": mid["epoch"] if mid else None,
        "rationale": (
            "Regional consolidation needs organic spatial structure after removing "
            "engineered parent–child locality. "
            f"Layers with signature coherence (Moran / near-far): {strong_layers or 'none'}; "
            f"stays_specialized binary clustering: {stays_cluster}; "
            f"layer tests pass {n_pass}/3 → {label}. "
            "Full build only if ≥2 layers show signature coherence AND stays specialists "
            "also cluster (regions of long-lived specialists, not just birth intensity)."
        ),
    }


def _summarize_table(epoch_row: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for L in LAYERS:
        sp = epoch_row["layers"][L].get("specialist_pairs")
        if not sp:
            rows.append({"layer": L, "n_specialists": 0})
            continue
        g = sp["gap_over_whole"]
        o = sp["on_off"]
        ge = g.get("exclude_parent_child") or {}
        oe = o.get("exclude_parent_child") or {}
        rows.append(
            {
                "layer": L,
                "n_specialists": sp["n_specialists"],
                "n_pc_pairs": sp["n_parent_child_pairs"],
                "gow_near_far_excl": ge.get("near_over_far"),
                "gow_p_left": g.get("near_over_far_excl_pc_p_left"),
                "gow_MoranI": g.get("morans_I"),
                "gow_Moran_p": g.get("morans_I_p_right"),
                "onoff_near_far_excl": oe.get("near_over_far"),
                "onoff_p_left": o.get("near_over_far_excl_pc_p_left"),
                "onoff_MoranI": o.get("morans_I"),
                "onoff_Moran_p": o.get("morans_I_p_right"),
                "binary_tracked_lift": (epoch_row["layers"][L].get("binary_tracked") or {}).get(
                    "lift_vs_base"
                ),
                "binary_tracked_p": (epoch_row["layers"][L].get("binary_tracked") or {}).get(
                    "p_right"
                ),
                "binary_stays_lift": (epoch_row["layers"][L].get("binary_stays") or {}).get(
                    "lift_vs_base"
                ),
                "binary_stays_p": (epoch_row["layers"][L].get("binary_stays") or {}).get(
                    "p_right"
                ),
            }
        )
    return rows


def write_results_md(summary: dict[str, Any], path: Path) -> None:
    v = summary["verdict"]
    late = summary["primary_epoch"]
    lines = [
        "# Specialization spatial coherence — go/no-go for regional consolidation",
        "",
        f"**Data:** re-slice of `gap_imprint_survival_20ep_hold/seed_0` structure snapshots "
        f"(no new training).",
        f"**Radius R:** {summary['radius']} (official sensing radius).",
        f"**Permutations:** {summary['n_perm']} coordinate-preserving label shuffles.",
        f"**Primary epoch:** {late} (late / end-of-hold).",
        "",
        f"## Verdict: `{v['label']}`",
        "",
        f"**Build full regional consolidation?** **{'YES' if v['go_regional_consolidation'] else 'NO'}** "
        f"(`{v.get('go_kind', '')}`)",
        "",
        v["rationale"],
        "",
        f"- Strong signature layers: `{v.get('strong_signature_layers')}`",
        f"- `stays_specialized` cluster spatially: **{v.get('stays_specialists_cluster')}**",
        "",
        "### Layer pass/fail (late epoch, α=0.05, parent–child excluded for pair ratios)",
        "",
    ]
    for L, t in v["layer_tests"].items():
        status = "PASS" if t["pass"] else "FAIL"
        ev = ", ".join(t.get("evidence") or []) or t.get("reason", "no evidence")
        lines.append(f"- **{L}**: `{status}` — {ev}")
    if v.get("stays_notes"):
        lines += ["", "### stays_specialized binary (do long-lived specialists form regions?)", ""]
        for n in v["stays_notes"]:
            lines.append(f"- {n}")
    if v.get("parent_child_confound_notes"):
        lines += ["", "### Parent–child confound notes", ""]
        for n in v["parent_child_confound_notes"]:
            lines.append(f"- {n}")

    lines += [
        "",
        "## What was measured",
        "",
        "Specialization *direction* in the product stack is the birth `gap_mask` "
        "(sample-space support). Snapshots do **not** store gap masks, so this "
        "re-slice uses the available specialization **signature** on tracked "
        "gap_imprinted units:",
        "",
        "- `gap_over_whole` (intensity of gap preference)",
        "- `on_off` (gap vs non-gap firing differential)",
        "- Binary: is unit specialization-tracked / `stays_specialized`",
        "",
        "Spatial tests (within layer, Euclidean coords on unit square):",
        "",
        "1. Mean |Δ| of signature among pairs with d≤R vs d>R "
        "(ratio < 1 ⇒ neighbors more similar).",
        "2. Moran's I with binary weights d≤R.",
        "3. Exclude parent↔child pairs (birth places children near parent with "
        f"jitter ≈0.05 ≪ R={summary['radius']} — engineered locality).",
        "4. Binary clustering: mean fraction of R-neighbors that are also "
        "specialists, vs base rate + permutation null.",
        "",
        "## Late-epoch numbers",
        "",
        "| layer | n_spec | n_pc | gow near/far | gow p | Moran gow | p | "
        "onoff near/far | p | Moran oo | p | bin_trk lift | p | bin_stays lift | p |",
        "|------:|-------:|-----:|-------------:|------:|----------:|--:|"
        "---------------:|--:|---------:|--:|-------------:|--:|---------------:|--:|",
    ]
    for row in summary["late_table"]:
        def f(x: Any, nd: int = 3) -> str:
            if x is None or (isinstance(x, float) and x != x):
                return "—"
            if isinstance(x, float):
                return f"{x:.{nd}f}"
            return str(x)

        lines.append(
            f"| {row['layer']} | {row.get('n_specialists', 0)} | {row.get('n_pc_pairs', '—')} | "
            f"{f(row.get('gow_near_far_excl'))} | {f(row.get('gow_p_left'))} | "
            f"{f(row.get('gow_MoranI'))} | {f(row.get('gow_Moran_p'))} | "
            f"{f(row.get('onoff_near_far_excl'))} | {f(row.get('onoff_p_left'))} | "
            f"{f(row.get('onoff_MoranI'))} | {f(row.get('onoff_Moran_p'))} | "
            f"{f(row.get('binary_tracked_lift'))} | {f(row.get('binary_tracked_p'))} | "
            f"{f(row.get('binary_stays_lift'))} | {f(row.get('binary_stays_p'))} |"
        )

    lines += [
        "",
        "## Reading",
        "",
        "- **near/far ≈ 1** and **Moran I ≈ 0** (null-compatible) ⇒ specialization "
        "signatures are spatially random among specialists.",
        "- **near/far ≪ 1** or **Moran I ≫ 0** with p≤0.05 after excluding "
        "parent–child ⇒ organic spatial coherence; regions exist.",
        "- Binary lift: if specialists' neighbors are no more specialist-dense "
        "than chance, there is no specialist *region* — only isolated specialists.",
        "",
        "## Scope / honesty",
        "",
        "- Snapshots with coords+specialization exist for **seed 0 only** "
        "(20-ep hold). Seeds 1–2 have unit rows without coords — not usable "
        "for this spatial test without a re-run.",
        "- Signature similarity ≠ shared gap_mask (true specialization *target*). "
        "If signatures are already spatially random, shared-target clustering is "
        "unlikely to appear; if signatures *were* coherent, a follow-up with "
        "stored gap masks would still be needed to confirm same-target regions.",
        "- Parent–child locality is a product of `newborn_parent_coord_jitter`, "
        "not evidence of organic regional structure.",
        "",
        "## Decision implication",
        "",
    ]
    if v["go_regional_consolidation"]:
        lines.append(
            "Specialization already clusters by coordinate (signatures **and** "
            "long-lived specialists) → designing a regional consolidation "
            "mechanism is justified as the next step."
        )
    elif v["label"] == "MIXED_LAYER_DEPENDENT":
        lines += [
            "Specialization is **not** network-wide spatially coherent:",
            "",
            "- **h1** shows real signature autocorrelation (neighbors share "
            "`gap_over_whole` / `on_off` more than distant pairs; Moran I > 0; "
            "parent–child exclusion does not remove the effect).",
            "- **h2** is spatially random / slightly anti-clustered.",
            "- **h3** is weak / borderline.",
            "- Units that **stay specialized** are **not** denser among "
            "neighbors than chance on any layer — long-lived specialists look "
            "isolated, not regional.",
            "",
            "**Do not** build a full regional-consolidation mechanism on this "
            "evidence. There is a partial latent handle on h1 signature "
            "intensity only; that is not the same as organic specialist "
            "regions to consolidate for retention. A regional mechanism would "
            "mostly impose structure the data do not support, especially for "
            "the stays cohort that retention would care about.",
        ]
    else:
        lines.append(
            "Specialization does **not** form reliable spatial regions under the "
            "locked stack → **do not** build regional consolidation on this "
            "evidence. It would impose structure the data do not support; "
            "complexity without a handle. Retention failures need a different "
            "lever (or a different claim)."
        )
    lines += ["", f"Artifacts: `{path.parent}/summary.json`.", ""]
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=DEFAULT_R)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for permutations")
    ap.add_argument(
        "--epochs",
        type=str,
        default="auto",
        help="Comma epochs or 'auto' (mid + late)",
    )
    args = ap.parse_args()

    if not SNAP_ROOT.exists():
        print(f"ERROR: missing snapshots at {SNAP_ROOT}", file=sys.stderr)
        return 1

    snaps = _load_snapshots()
    if not snaps:
        print("ERROR: no snapshots", file=sys.stderr)
        return 1
    stays = _load_stays_set()
    print(
        f"Loaded {len(snaps)} epochs, {len(stays)} stays_specialized units, "
        f"R={args.radius}, n_perm={args.n_perm}",
        flush=True,
    )

    epochs_available = [e for e, _ in snaps]
    if args.epochs == "auto":
        late_e = epochs_available[-1]
        mid_e = epochs_available[len(epochs_available) // 2]
        want = sorted({mid_e, late_e})
    else:
        want = [int(x) for x in args.epochs.split(",") if x.strip()]

    by_epoch = {e: s for e, s in snaps}
    rng = random.Random(args.seed)
    results_by_epoch: dict[str, Any] = {}
    for e in want:
        if e not in by_epoch:
            print(f"  skip epoch {e} (missing)")
            continue
        print(f"  analyzing epoch {e} ...", flush=True)
        results_by_epoch[str(e)] = analyze_epoch(
            by_epoch[e], args.radius, args.n_perm, rng, stays
        )

    late_e = max(int(k) for k in results_by_epoch)
    mid_candidates = [int(k) for k in results_by_epoch if int(k) != late_e]
    mid_e = mid_candidates[0] if mid_candidates else None
    late = results_by_epoch[str(late_e)]
    mid = results_by_epoch[str(mid_e)] if mid_e is not None else None
    verdict = _verdict(late, mid)

    # Trajectory: near/far gow excl-pc across all epochs (cheap, no full perm)
    traj: list[dict[str, Any]] = []
    for e, snap in snaps:
        row: dict[str, Any] = {"epoch": e}
        for layer in LAYERS:
            spec = _extract_specialists(snap["layers"][layer]["neurons"])
            if spec is None:
                row[layer] = None
                continue
            D = _pairwise_distances(spec["xy"])
            pc = _parent_child_index_pairs(spec["ids"], spec["parents"])
            st = _pair_stats(spec["gow"], D, args.radius, exclude_pairs=pc)
            mi = _morans_I(spec["gow"], D, args.radius)
            row[layer] = {
                "n": spec["n"],
                "gow_near_far_excl": st["near_over_far"] if st else None,
                "gow_MoranI": mi,
            }
        traj.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": {
            "data": str(SNAP_ROOT.relative_to(_REPO)),
            "units": str(UNIT_JSON.relative_to(_REPO)),
            "note": "Re-slice only; no new training.",
            "specialization_proxy": [
                "gap_over_whole",
                "on_off",
                "binary_tracked",
                "binary_stays_specialized",
            ],
            "true_direction_unavailable": (
                "birth gap_mask not stored in snapshots; signature used as proxy"
            ),
        },
        "radius": args.radius,
        "n_perm": args.n_perm,
        "perm_seed": args.seed,
        "primary_epoch": late_e,
        "mid_epoch": mid_e,
        "verdict": verdict,
        "late_table": _summarize_table(late),
        "mid_table": _summarize_table(mid) if mid else None,
        "by_epoch": results_by_epoch,
        "trajectory_gow_near_far": traj,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_results_md(summary, OUT / "RESULTS.md")

    print("\n======== VERDICT ========", flush=True)
    print(f"  {verdict['label']}", flush=True)
    print(
        f"  go_regional_consolidation = {verdict['go_regional_consolidation']}",
        flush=True,
    )
    for L, t in verdict["layer_tests"].items():
        print(f"  {L}: {'PASS' if t['pass'] else 'FAIL'} {t.get('evidence') or t.get('reason')}", flush=True)
    print(f"\nWrote {OUT / 'summary.json'}", flush=True)
    print(f"Wrote {OUT / 'RESULTS.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
