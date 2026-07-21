#!/usr/bin/env python3
"""Render Phase 1a structure snapshots (static figure or multi-epoch animation).

Expects full-state dumps from ``record_structure_snapshots: true`` under a
results root (``structure_snapshots/epXXXX_*.json``). Does not reconstruct
topology from aggregate ``structure_timeseries.csv``.

Examples:
  python scripts/visualize_structure.py \\
      --results results/phase1a_diagnostics/visualizations/example_seed0 \\
      --tag late -o results/phase1a_diagnostics/visualizations/late.png

  python scripts/visualize_structure.py \\
      --results results/phase1a_diagnostics/visualizations/example_seed0 \\
      --animate -o results/phase1a_diagnostics/visualizations/structure.gif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from neuronca.plasticity.structure_snapshot import (  # noqa: E402
    list_structure_snapshots,
    load_structure_snapshot,
)

# Cap on *init_dense* edges drawn per subplot for legibility.
# gap_imprinted_seeded and incentive_grown are ALWAYS drawn in full (never
# subsampled into the init budget — the color distinction is the point).
DEFAULT_MAX_INIT_EDGES_PER_SUBPLOT = 500
# Legacy alias used by CLI / callers.
DEFAULT_MAX_EDGES_PER_SUBPLOT = DEFAULT_MAX_INIT_EDGES_PER_SUBPLOT

# Pixel density: 2.5× prior defaults (was dpi=140 static / 100 anim).
STATIC_DPI = 350
ANIM_DPI = 250
# Extra vertical room: suptitle (top) + multi-line ax titles + bottom legend.
FIGSIZE = (14.0, 5.8)
# Fixed margins so animation frames do not jump when titles change length.
# (Avoid bbox_inches="tight" on frames — it reflows and collides with legend.)
SUBPLOT_ADJUST = dict(left=0.04, right=0.99, top=0.78, bottom=0.16, wspace=0.22)

# Colors by birth / edge origin.
NEURON_COLORS = {
    "init": "#9aa0a6",
    "kaiming": "#5c6bc0",
    "parent_perturbed": "#26a69a",
    "gap_imprinted": "#e53935",
    "grown": "#5c6bc0",
    "none": "#bdbdbd",
}
EDGE_STYLES = {
    "init_dense": {"color": "#b0b0b0", "alpha": 0.08, "lw": 0.3},
    "incentive_grown": {"color": "#1e88e5", "alpha": 0.55, "lw": 0.7},
    "gap_imprinted_seeded": {"color": "#e53935", "alpha": 0.75, "lw": 0.9},
}
LAYER_ORDER = ("h1", "h2", "h3")
# Linear layers drawn in each subplot (edges *into* that hidden layer).
LAYER_INCOMING = {"h1": "fc1", "h2": "fc2", "h3": "fc3"}


def _resolve_snapshot_path(
    results: Path,
    *,
    epoch: int | None,
    tag: str | None,
) -> Path:
    entries = list_structure_snapshots(results)
    if not entries:
        raise FileNotFoundError(
            f"no structure_snapshots under {results}; "
            "re-run with experiment.record_structure_snapshots: true"
        )
    if tag is not None:
        matches = [e for e in entries if str(e.get("tag")) == tag]
        if not matches:
            available = sorted({str(e.get("tag")) for e in entries})
            raise FileNotFoundError(
                f"tag {tag!r} not found; available: {available}"
            )
        return Path(matches[-1]["path"])
    if epoch is not None:
        matches = [e for e in entries if int(e.get("epoch", -1)) == int(epoch)]
        if not matches:
            available = sorted({int(e.get("epoch", -1)) for e in entries})
            raise FileNotFoundError(
                f"epoch {epoch} not found; available: {available}"
            )
        return Path(matches[-1]["path"])
    # Default: last snapshot.
    return Path(entries[-1]["path"])


def _subsample_edges(
    edges: list[dict[str, Any]],
    max_init_edges: int,
    *,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Always keep all grown/gap edges; subsample only init_dense background.

    Policy (explicit): the ``max_init_edges`` cap applies **only** to
    ``init_dense``. ``gap_imprinted_seeded`` and ``incentive_grown`` are
    never dropped by this cap — even when their count alone exceeds the
    numeric budget. Total drawn can therefore exceed ``max_init_edges``.

    Returns (selected, meta).
    """
    gap = [e for e in edges if e.get("origin") == "gap_imprinted_seeded"]
    incentive = [e for e in edges if e.get("origin") == "incentive_grown"]
    init = [e for e in edges if e.get("origin") == "init_dense"]
    other = [
        e
        for e in edges
        if e.get("origin")
        not in ("incentive_grown", "gap_imprinted_seeded", "init_dense")
    ]
    # Always keep interesting edges in full.
    keep: list[dict[str, Any]] = list(gap) + list(incentive) + list(other)
    n_init_drop = 0
    if len(init) > max_init_edges:
        idx = rng.choice(len(init), size=int(max_init_edges), replace=False)
        keep.extend(init[i] for i in sorted(idx))
        n_init_drop = len(init) - int(max_init_edges)
    else:
        keep.extend(init)
    meta = {
        "total": len(edges),
        "drawn": len(keep),
        "dropped": n_init_drop,
        "dropped_init_dense_only": n_init_drop,
        "kept_gap_imprinted_seeded": len(gap),
        "kept_incentive_grown": len(incentive),
        "kept_init_dense": len(init) - n_init_drop,
        "max_init_edges": int(max_init_edges),
        "policy": "never_drop_gap_or_incentive; subsample_init_dense_only",
    }
    return keep, meta


def _coord_lookup(
    snap: dict[str, Any],
    pop: str,
) -> dict[int, tuple[float, float]]:
    """Map unit id → (x, y). Hidden from snapshot; in/out are grid / class line."""
    if pop in LAYER_ORDER:
        neurons = (snap.get("layers") or {}).get(pop, {}).get("neurons") or []
        return {
            int(n["id"]): (float(n["x"]), float(n["y"]))
            for n in neurons
            if "x" in n and "y" in n
        }
    if pop == "in":
        # 28×28 MNIST pixel grid (matches input_coords_28x28).
        out: dict[int, tuple[float, float]] = {}
        for i in range(784):
            row, col = divmod(i, 28)
            out[i] = (col / 27.0, row / 27.0)
        return out
    if pop == "out":
        n = 10
        return {i: (i / max(n - 1, 1), 0.0) for i in range(n)}
    return {}


def _size_for_neuron(n: dict[str, Any], base: float = 18.0) -> float:
    """Point size from activation EMA or gap/whole when present."""
    gow = n.get("gap_over_whole")
    if gow is not None and gow == gow:
        return float(np.clip(base * (0.6 + 0.5 * float(gow)), 8.0, 80.0))
    ema = n.get("activation_ema")
    if ema is not None and ema == ema:
        return float(np.clip(base * (0.5 + float(ema)), 8.0, 60.0))
    return base


def render_structure_figure(
    snap: dict[str, Any],
    *,
    max_edges: int = DEFAULT_MAX_INIT_EDGES_PER_SUBPLOT,
    show_parent_links: bool = True,
    seed: int = 0,
    title: str | None = None,
    dying_ids: set[tuple[str, int]] | None = None,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Draw one figure with one subplot per hidden layer.

    ``dying_ids``: (layer, slot) marked on this frame as about-to-die / last
    appearance (used by the animator after comparing consecutive snapshots).
    Pruned units with coords in ``snap['pruned_neurons']`` are drawn as open
    red circles on the prune frame (they are no longer in the active list).
    """
    rng = np.random.default_rng(seed)
    # Fixed layout (not constrained_layout): keeps title/legend bands stable
    # across frames when edge counts grow and title text gets longer.
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    fig.subplots_adjust(**SUBPLOT_ADJUST)
    meta: dict[str, Any] = {"edge_subsample": {}}
    dying_ids = dying_ids or set()
    pruned_by_layer: dict[str, list[dict[str, Any]]] = {h: [] for h in LAYER_ORDER}
    for p in snap.get("pruned_neurons") or []:
        layer = str(p.get("layer", ""))
        if layer in pruned_by_layer:
            pruned_by_layer[layer].append(p)

    for ax, layer in zip(axes, LAYER_ORDER):
        neurons = (snap.get("layers") or {}).get(layer, {}).get("neurons") or []
        lin = LAYER_INCOMING[layer]
        edges = list((snap.get("edges") or {}).get(lin) or [])
        selected, sub_meta = _subsample_edges(edges, max_edges, rng=rng)
        sub_meta["linear"] = lin
        meta["edge_subsample"][layer] = sub_meta
        n_tot = int(sub_meta["total"])
        n_drop = int(sub_meta["dropped"])
        n_gap_kept = int(sub_meta["kept_gap_imprinted_seeded"])

        post_coords = _coord_lookup(snap, layer)
        # Incoming pre population coords.
        pre_pop = {"h1": "in", "h2": "h1", "h3": "h2"}[layer]
        pre_coords = _coord_lookup(snap, pre_pop)

        # Draw edges (init low opacity first, then grown).
        segs_by_origin: dict[str, list[list[tuple[float, float]]]] = {
            "init_dense": [],
            "incentive_grown": [],
            "gap_imprinted_seeded": [],
        }
        for e in selected:
            origin = str(e.get("origin", "init_dense"))
            i, j = int(e["i"]), int(e["j"])
            if i not in pre_coords or j not in post_coords:
                continue
            p0 = pre_coords[i]
            p1 = post_coords[j]
            bucket = origin if origin in segs_by_origin else "incentive_grown"
            segs_by_origin[bucket].append([p0, p1])

        for origin in ("init_dense", "incentive_grown", "gap_imprinted_seeded"):
            segs = segs_by_origin[origin]
            if not segs:
                continue
            style = EDGE_STYLES[origin]
            lc = LineCollection(
                segs,
                colors=style["color"],
                alpha=style["alpha"],
                linewidths=style["lw"],
                zorder=1 if origin == "init_dense" else 2,
            )
            ax.add_collection(lc)

        # Parent-locus links for gap_imprinted / parent_perturbed children.
        if show_parent_links:
            parent_segs: list[list[tuple[float, float]]] = []
            for n in neurons:
                parent = n.get("parent_slot")
                origin = str(n.get("origin", n.get("birth_kind", "")))
                if parent is None or origin not in (
                    "gap_imprinted",
                    "parent_perturbed",
                ):
                    continue
                child_id = int(n["id"])
                parent = int(parent)
                if child_id not in post_coords or parent not in post_coords:
                    continue
                parent_segs.append([post_coords[parent], post_coords[child_id]])
            if parent_segs:
                lc = LineCollection(
                    parent_segs,
                    colors="#ff8f00",
                    alpha=0.85,
                    linewidths=1.1,
                    linestyles="--",
                    zorder=3,
                )
                ax.add_collection(lc)

        # Active neurons.
        for n in neurons:
            nid = int(n["id"])
            x, y = float(n["x"]), float(n["y"])
            origin = str(n.get("origin", n.get("birth_kind", "init")))
            color = NEURON_COLORS.get(origin, "#757575")
            size = _size_for_neuron(n)
            z = 5
            edgecolor = "none"
            lw = 0.0
            if (layer, nid) in dying_ids:
                # Last-live-frame highlight (unit present now, absent next).
                edgecolor = "#d50000"
                lw = 2.0
                z = 6
            elif n.get("grown_this_step"):
                edgecolor = "#00c853"
                lw = 1.2
                z = 6
            ax.scatter(
                [x],
                [y],
                s=size,
                c=[color],
                edgecolors=edgecolor,
                linewidths=lw,
                zorder=z,
                alpha=0.95,
            )

        # Death marks: open red circles at last-living coords (prune frame).
        n_pruned_drawn = 0
        for p in pruned_by_layer[layer]:
            x, y = p.get("x"), p.get("y")
            if x is None or y is None:
                continue
            try:
                xf, yf = float(x), float(y)
            except (TypeError, ValueError):
                continue
            if not (xf == xf and yf == yf):  # NaN guard
                continue
            ax.scatter(
                [xf],
                [yf],
                s=70,
                facecolors="none",
                edgecolors="#d50000",
                linewidths=2.2,
                zorder=7,
                alpha=1.0,
            )
            n_pruned_drawn += 1
        meta["edge_subsample"][layer]["pruned_markers_drawn"] = n_pruned_drawn

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        # Compact multi-line title — never one long line that spills into neighbors.
        title_lines = [
            layer,
            f"n={len(neurons)}  gap_e={n_gap_kept}",
            f"edges {len(selected)}/{n_tot}"
            + (f"  (−{n_drop} init)" if n_drop else "")
            + (f"  die={n_pruned_drawn}" if n_pruned_drawn else ""),
        ]
        ax.set_title("\n".join(title_lines), fontsize=9, pad=8, linespacing=1.15)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_alpha(0.3)

    legend_elems = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=NEURON_COLORS["init"],
            markersize=8,
            label="init neuron",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=NEURON_COLORS["gap_imprinted"],
            markersize=8,
            label="gap_imprinted",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=NEURON_COLORS["kaiming"],
            markersize=8,
            label="kaiming / grown",
        ),
        Line2D(
            [0],
            [0],
            color=EDGE_STYLES["init_dense"]["color"],
            alpha=0.5,
            lw=1.5,
            label="init_dense edge",
        ),
        Line2D(
            [0],
            [0],
            color=EDGE_STYLES["gap_imprinted_seeded"]["color"],
            lw=1.5,
            label="gap-seeded edge",
        ),
        Line2D(
            [0],
            [0],
            color="#ff8f00",
            ls="--",
            lw=1.5,
            label="parent locus",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="white",
            markeredgecolor="#d50000",
            markersize=8,
            markeredgewidth=1.5,
            label="pruned (last frame)",
        ),
    ]
    # Legend sits in the reserved bottom band (below axes), never over plots.
    fig.legend(
        handles=legend_elems,
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        columnspacing=1.2,
        handletextpad=0.4,
        borderaxespad=0.0,
    )
    ep = snap.get("epoch")
    tag = snap.get("tag")
    # Suptitle in the reserved top band; keep short.
    if title is None:
        title = f"Structure  ·  epoch={ep}  ·  tag={tag}  ·  step={snap.get('plasticity_step')}"
    fig.suptitle(title, fontsize=11, y=0.97)
    return fig, meta


def _active_ids(snap: dict[str, Any]) -> set[tuple[str, int]]:
    out: set[tuple[str, int]] = set()
    for layer in LAYER_ORDER:
        for n in (snap.get("layers") or {}).get(layer, {}).get("neurons") or []:
            out.add((layer, int(n["id"])))
    return out


def render_static(
    results: Path,
    out_path: Path,
    *,
    epoch: int | None = None,
    tag: str | None = None,
    max_edges: int = DEFAULT_MAX_EDGES_PER_SUBPLOT,
    show_parent_links: bool = True,
) -> Path:
    path = _resolve_snapshot_path(results, epoch=epoch, tag=tag)
    snap = load_structure_snapshot(path)
    # Prefer pruned_this_step from the dump if present.
    dying = {
        (str(p["layer"]), int(p["slot"]))
        for p in (snap.get("pruned_this_step") or [])
    }
    fig, meta = render_structure_figure(
        snap,
        max_edges=max_edges,
        show_parent_links=show_parent_links,
        dying_ids=dying,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # pad_inches only — do not use bbox_inches="tight" (reflows legend into axes).
    fig.savefig(out_path, dpi=STATIC_DPI, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    # Log subsample info next to the figure for audit.
    log_path = out_path.with_suffix(out_path.suffix + ".edge_subsample.json")
    import json

    log_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out_path} (dpi={STATIC_DPI})")
    for layer, info in meta["edge_subsample"].items():
        print(
            f"  {layer}: drawn {info['drawn']}/{info['total']} "
            f"(gap={info.get('kept_gap_imprinted_seeded', '?')}, "
            f"init_drop={info.get('dropped', 0)}, "
            f"die_marks={info.get('pruned_markers_drawn', 0)})"
        )
    return out_path


def render_animation(
    results: Path,
    out_path: Path,
    *,
    max_edges: int = DEFAULT_MAX_INIT_EDGES_PER_SUBPLOT,
    show_parent_links: bool = True,
    fps: float = 1.5,
    dpi: int = ANIM_DPI,
) -> Path:
    entries = list_structure_snapshots(results)
    if not entries:
        raise FileNotFoundError(f"no structure_snapshots under {results}")
    snaps = [load_structure_snapshot(e["path"]) for e in entries]
    if len(snaps) < 1:
        raise FileNotFoundError("empty snapshot list")

    # Precompute dying sets: units present in frame t but not in t+1,
    # union snapshot-reported pruned_this_step.
    dying_per_frame: list[set[tuple[str, int]]] = []
    for i, snap in enumerate(snaps):
        dying = {
            (str(p["layer"]), int(p["slot"]))
            for p in (snap.get("pruned_this_step") or [])
        }
        if i + 1 < len(snaps):
            cur = _active_ids(snap)
            nxt = _active_ids(snaps[i + 1])
            dying |= cur - nxt
        dying_per_frame.append(dying)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Render frames to a temporary image list via FuncAnimation writers, or
    # save sequential PNGs and assemble. Prefer imageio if available for GIF.
    frames_dir = out_path.parent / f"_frames_{out_path.stem}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    for i, snap in enumerate(snaps):
        fig, _meta = render_structure_figure(
            snap,
            max_edges=max_edges,
            show_parent_links=show_parent_links,
            dying_ids=dying_per_frame[i],
            title=(
                f"Structure  epoch={snap.get('epoch')}  "
                f"tag={snap.get('tag')}  ({i + 1}/{len(snaps)})"
            ),
        )
        fp = frames_dir / f"frame_{i:04d}.png"
        # Fixed canvas (no tight crop) so every frame is the same size and the
        # legend band never slides up into the plots as titles grow.
        fig.savefig(fp, dpi=dpi, bbox_inches=None, pad_inches=0.15)
        plt.close(fig)
        frame_paths.append(fp)

    suffix = out_path.suffix.lower()
    if suffix in (".gif",):
        _write_gif(frame_paths, out_path, fps=fps)
    elif suffix in (".mp4", ".avi"):
        _write_mp4(frame_paths, out_path, fps=fps)
    else:
        # Default to GIF if no recognized extension.
        if not suffix:
            out_path = out_path.with_suffix(".gif")
        _write_gif(frame_paths, out_path, fps=fps)

    print(f"wrote {out_path} ({len(frame_paths)} frames)")
    return out_path


def _write_gif(frame_paths: list[Path], out_path: Path, *, fps: float) -> None:
    try:
        import imageio.v2 as imageio  # type: ignore

        images = [imageio.imread(str(p)) for p in frame_paths]
        duration = 1.0 / max(fps, 1e-3)
        imageio.mimsave(str(out_path), images, duration=duration, loop=0)
        return
    except ImportError:
        pass
    # Fallback: Pillow
    try:
        from PIL import Image

        imgs = [Image.open(p).convert("RGB") for p in frame_paths]
        duration_ms = int(1000.0 / max(fps, 1e-3))
        imgs[0].save(
            out_path,
            save_all=True,
            append_images=imgs[1:],
            duration=duration_ms,
            loop=0,
        )
        return
    except ImportError as exc:
        raise RuntimeError(
            "need imageio or Pillow to write GIF animations"
        ) from exc


def _write_mp4(frame_paths: list[Path], out_path: Path, *, fps: float) -> None:
    try:
        import imageio.v2 as imageio  # type: ignore

        writer = imageio.get_writer(str(out_path), fps=fps)
        for p in frame_paths:
            writer.append_data(imageio.imread(str(p)))
        writer.close()
        return
    except Exception:
        pass
    # Fallback via matplotlib animation + ffmpeg
    try:
        import matplotlib.animation as animation
        from matplotlib.image import imread

        fig, ax = plt.subplots(figsize=(12.5, 4.2))
        ax.axis("off")
        im0 = imread(str(frame_paths[0]))
        artist = ax.imshow(im0)

        def _update(i: int):
            artist.set_data(imread(str(frame_paths[i])))
            return (artist,)

        anim = animation.FuncAnimation(
            fig, _update, frames=len(frame_paths), interval=1000.0 / max(fps, 1e-3)
        )
        anim.save(str(out_path), fps=fps, dpi=100)
        plt.close(fig)
    except Exception as exc:
        raise RuntimeError(
            f"could not write MP4 to {out_path}; install imageio[ffmpeg] "
            f"or use --animate with a .gif path ({exc})"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Seed / results root containing structure_snapshots/",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output figure path (.png/.gif/.mp4)",
    )
    p.add_argument("--epoch", type=int, default=None, help="0-based epoch index")
    p.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Snapshot tag (e.g. early / mid / late / epoch_3)",
    )
    p.add_argument(
        "--animate",
        action="store_true",
        help="Render all snapshots as an animation (GIF/MP4)",
    )
    p.add_argument(
        "--max-edges",
        type=int,
        default=DEFAULT_MAX_INIT_EDGES_PER_SUBPLOT,
        help=(
            "Max init_dense edges per subplot. gap_imprinted_seeded and "
            "incentive_grown are always drawn in full (never capped)."
        ),
    )
    p.add_argument(
        "--no-parent-links",
        action="store_true",
        help="Disable child→parent locus dashed lines",
    )
    p.add_argument("--fps", type=float, default=1.5)
    args = p.parse_args(argv)

    results = args.results
    if args.animate:
        out = args.output or (
            results / "structure_animation.gif"
            if results.is_dir()
            else Path("structure_animation.gif")
        )
        render_animation(
            results,
            out,
            max_edges=args.max_edges,
            show_parent_links=not args.no_parent_links,
            fps=args.fps,
        )
    else:
        out = args.output or (
            results / "structure_static.png"
            if results.is_dir()
            else Path("structure_static.png")
        )
        render_static(
            results,
            out,
            epoch=args.epoch,
            tag=args.tag,
            max_edges=args.max_edges,
            show_parent_links=not args.no_parent_links,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
