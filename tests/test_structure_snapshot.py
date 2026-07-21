"""Structure snapshot export + cadence (observability only)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from neuronca.models.dynamic_mlp import DynamicMLPReasoningNet
from neuronca.plasticity.masks import (
    EDGE_ORIGIN_GAP_IMPRINT,
    NEURON_BIRTH_GAP_IMPRINTED,
    NEURON_BIRTH_INIT,
)
from neuronca.plasticity.structure_snapshot import (
    StructureSnapshotRecorder,
    capture_structure_snapshot,
    list_structure_snapshots,
    resolve_snapshot_epochs,
    write_structure_snapshot,
)
from neuronca.signals.activation import ActivationMagnitude
from neuronca.broadcast.error import BroadcastPacket


def test_resolve_snapshot_epochs_every_and_early_mid_late():
    every = resolve_snapshot_epochs(6, "every_epoch")
    assert set(every.keys()) == set(range(6))
    eml = resolve_snapshot_epochs(6, "early_mid_late")
    # 1-based early=1→0, mid=3→2, late=6→5
    assert eml[0] == "early"
    assert eml[2] == "mid"
    assert eml[5] == "late"
    listed = resolve_snapshot_epochs(10, [0, 4, 9])
    assert listed == {0: "epoch_0", 4: "epoch_4", 9: "epoch_9"}


def test_capture_includes_init_neurons_and_edges(tmp_path: Path):
    net = DynamicMLPReasoningNet(
        hidden_init=(4, 3, 2),
        hidden_max=(8, 6, 4),
        coord_seed=0,
    )
    signal = ActivationMagnitude(decay=0.99)
    # Warm signal buffers with one forward.
    x = torch.randn(8, 784)
    logits = net(x)
    assert logits.shape == (8, 10)
    acts = net.last_activations
    assert acts is not None
    signal.update(
        activations=acts,
        layer_shapes=net.layer_shapes(),
        broadcast=BroadcastPacket(
            e_scalar_ce=0.0,
            e_vec_mean=torch.zeros(10),
            e_s_per_sample=torch.zeros(8),
            r_per_sample=torch.zeros(8, 10),
            step=0,
        ),
        step=0,
    )
    snap = capture_structure_snapshot(
        net, epoch=0, tag="epoch_0", signal=signal
    )
    assert snap["schema"].startswith("neuronca.structure_snapshot")
    assert len(snap["layers"]["h1"]["neurons"]) == 4
    n0 = snap["layers"]["h1"]["neurons"][0]
    assert n0["origin"] == "init"
    assert n0["birth_kind"] == "init"
    assert "x" in n0 and "y" in n0
    assert n0["activation_ema"] is not None
    # Dense init edges present on fc1.
    assert len(snap["edges"]["fc1"]) == 4 * 784
    assert all(e["origin"] == "init_dense" for e in snap["edges"]["fc1"][:10])

    path = write_structure_snapshot(tmp_path, snap)
    assert path.is_file()
    entries = list_structure_snapshots(tmp_path)
    assert len(entries) == 1
    assert entries[0]["tag"] == "epoch_0"


def test_gap_imprint_provenance_in_snapshot():
    net = DynamicMLPReasoningNet(
        hidden_init=(4, 3, 2),
        hidden_max=(8, 6, 4),
        coord_seed=1,
    )
    net.begin_plasticity_step()
    # Gap imprint into free slot with a toy residual direction.
    fan_in = net.fc1.in_features
    row = torch.randn(fan_in)
    info = net.activate_neuron(
        "h1",
        slot=None,
        weight_init="gap_imprinted",
        parent_slot=0,
        gap_weight_row=row,
        parent_noise=0.0,
        birth_bias=0.0,
    )
    slot = int(info["slot"])
    assert int(net.birth_kind("h1")[slot].item()) == NEURON_BIRTH_GAP_IMPRINTED
    assert int(net.parent_slot("h1")[slot].item()) == 0
    # Seed one gap edge manually for origin tag check.
    free_in = 10
    net.add_edge("fc1", slot, free_in, origin=EDGE_ORIGIN_GAP_IMPRINT, reinit_weight=True)
    snap = capture_structure_snapshot(net, epoch=1, tag="mid")
    grown = [n for n in snap["layers"]["h1"]["neurons"] if n["id"] == slot][0]
    assert grown["origin"] == "gap_imprinted"
    assert grown["parent_slot"] == 0
    gap_edges = [
        e for e in snap["edges"]["fc1"] if e["origin"] == "gap_imprinted_seeded"
    ]
    assert any(e["j"] == slot and e["i"] == free_in for e in gap_edges)


def test_recorder_writes_only_selected_epochs(tmp_path: Path):
    net = DynamicMLPReasoningNet(
        hidden_init=(4, 3, 2),
        hidden_max=(8, 6, 4),
        coord_seed=0,
    )
    rec = StructureSnapshotRecorder.from_config(
        tmp_path, total_epochs=4, record=True, cadence=[1, 3]
    )
    assert rec is not None
    assert rec.maybe_record(0, net) is None
    p1 = rec.maybe_record(1, net)
    assert p1 is not None and p1.is_file()
    assert rec.maybe_record(2, net) is None
    p3 = rec.maybe_record(3, net)
    assert p3 is not None
    assert len(list_structure_snapshots(tmp_path)) == 2


def test_init_birth_kind_on_construction():
    net = DynamicMLPReasoningNet(
        hidden_init=(4, 3, 2),
        hidden_max=(8, 6, 4),
        coord_seed=0,
    )
    assert int(net.birth_kind("h1")[0].item()) == NEURON_BIRTH_INIT
    assert int(net.parent_slot("h1")[0].item()) == -1


def test_deactivate_returns_last_coords_and_snapshot_pruned_neurons():
    net = DynamicMLPReasoningNet(
        hidden_init=(4, 3, 2),
        hidden_max=(8, 6, 4),
        coord_seed=0,
    )
    net.begin_plasticity_step()
    before = net.coords("h1")[0].detach().clone()
    ev = net.deactivate_neuron("h1", 0)
    assert abs(float(ev["x"]) - float(before[0])) < 1e-5
    assert abs(float(ev["y"]) - float(before[1])) < 1e-5
    # Capture via prune records (as orchestrator would provide).
    snap = capture_structure_snapshot(
        net,
        epoch=2,
        tag="epoch_2",
        pruned_records=[
            {
                "layer": "h1",
                "slot": 0,
                "x": ev["x"],
                "y": ev["y"],
                "neuron_origin": ev["neuron_origin"],
                "birth_kind": ev["birth_kind"],
                "parent_slot": ev["parent_slot"],
                "birth_step": ev["birth_step"],
            }
        ],
    )
    assert len(snap["pruned_neurons"]) == 1
    pn = snap["pruned_neurons"][0]
    assert pn["id"] == 0 and pn["layer"] == "h1"
    assert pn["x"] == ev["x"] and pn["y"] == ev["y"]
    # Dead unit is not in the active neuron list.
    assert all(n["id"] != 0 for n in snap["layers"]["h1"]["neurons"])


def test_edge_subsample_never_drops_gap_or_incentive():
    """Renderer policy: cap applies to init_dense only."""
    import numpy as np

    # Import from scripts path via sys — reimplement check on pure function.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from visualize_structure import _subsample_edges

    edges = (
        [{"origin": "init_dense", "i": i, "j": 0} for i in range(2000)]
        + [{"origin": "gap_imprinted_seeded", "i": i, "j": 1} for i in range(300)]
        + [{"origin": "incentive_grown", "i": i, "j": 2} for i in range(250)]
    )
    rng = np.random.default_rng(0)
    selected, meta = _subsample_edges(edges, max_init_edges=100, rng=rng)
    assert meta["kept_gap_imprinted_seeded"] == 300
    assert meta["kept_incentive_grown"] == 250
    assert meta["kept_init_dense"] == 100
    assert meta["dropped"] == 1900
    assert meta["policy"].startswith("never_drop_gap")
    gap_in_sel = sum(1 for e in selected if e["origin"] == "gap_imprinted_seeded")
    inc_in_sel = sum(1 for e in selected if e["origin"] == "incentive_grown")
    assert gap_in_sel == 300
    assert inc_in_sel == 250
    # Total drawn = 300 + 250 + 100 > max_init_edges (soft cap).
    assert len(selected) == 650
