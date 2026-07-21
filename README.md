# NeuroNCA

Research codebase for **structural plasticity** experiments. The reasoning network
always trains with standard **backpropagation**. Local importance signals are
**observed alongside** training; in Phase 1a they also drive **homeostatic grow/prune**
of hidden neurons and adjacent edges (never a topology optimizer over multi-hop graphs).

## Current status: Phase 0 complete · Phase 1a mechanism closed on MNIST · Fashion recalibrating

| Phase | Status |
|-------|--------|
| **Phase 0** | Complete — local signal validation (`activation` = best current-task proxy) |
| **Phase 1a** | **MNIST mechanism + MNIST Pareto closed**; Fashion-MNIST **must use dataset-calibrated r\*_ℓ** before efficiency claims |

**Read first for Phase 1a:**  
[`docs/phase1a_results.md`](docs/phase1a_results.md) ·  
[`docs/phase1a_preregistration.md`](docs/phase1a_preregistration.md)

---

## Phase 1a in one page

**Scope:** Fixed layer layout (input → H1 → H2 → H3 → output). Plastic **hidden neuron
counts** and **adjacent inter-layer edges** only. Locality = sensing radius `R` in a
shared unit square. Dense Phase 0 start (prune-first dynamics).

**Official stack** (`configs/phase1a_fixed.yaml`):

| Piece | Setting |
|-------|---------|
| Threshold | **fixed** (sliding tested long-horizon; product default still fixed) |
| Growth trigger | **gated novelty** (novelty ∧ local CE ∧ \|N\| ≥ **k_min=11**) |
| Birth | **gap_imprinted** (sep-power edges, open-fan Kaiming, straddle bias) |
| Prune | continuous competitive z-score / fraction |
| Grace | 3 steps (≥ birth-bias steps) |

### Mechanism (specialization)

| Horizon | Establish of births | Stays specialized \| est∧gate |
|---------|--------------------:|------------------------------:|
| 6 ep (primary A/B) | ~53% | ~**91–92%** (sliding ≈ fixed, `B_NO_HELP`) |
| 20 ep fixed hold | ~**85%** | ~**79%** |
| 20 ep early-cohort curve | — | still-stays falls with age (~0.95→0.72 by age 18) |
| 20 ep sliding retest | — | early erosion **reduced** (`B_REDUCES_EARLY_EROSION`) |

Establishment and stays are real under the locked stack. Early-cohort erosion under
fixed is progressive (not “early births are worse” — matched age early≈late). Sliding
helps durability; 6-ep `B_NO_HELP` was incomplete for that claim.

### Efficiency (Pareto vs dense fixed-topology)

Pre-registered post-hoc Pareto: plastic final size is **endogenous**; dense baselines
train on a size grid; pass bar **δ = 1.0 pp** test acc (`PARETO_ACC_AT_COST_*`).

| Dataset | Result | Notes |
|---------|--------|--------|
| **MNIST** | **`PARETO_BOTH_FAIL`** | Dense already flat/high by ~81k params; plastic ~96.6% @ ~88k vs dense ~97.9% @ 81k. Fixed near-boundary (−0.23 pp); dense seed ranges ~0.12 pp median — dominance not noise. |
| **Fashion-MNIST** | **recalibration first** | Dense scout still valid (flat high-80s from ~53k). **Prior plastic Pareto used MNIST r\*_ℓ — INVALID.** Fashion probe locks **r\* = (0.71, 0.80, 0.91)**; straddle OK; growth-fire PASS. Recalibrated 20-ep plastic Pareto not yet the headline. |

**Product framing:** MNIST mechanism specialization stands; MNIST efficiency vs dense is not competitive under this stack/horizon. **Fashion efficiency claims require Fashion-calibrated homeostatic targets** (every `r*_ℓ` was MNIST-derived; reusing them is not a transfer test). Do not present 20-ep numbers as long-run equilibrium (specialists still mid-decline under fixed).

### Key artifacts

| What | Path |
|------|------|
| Full write-up | `docs/phase1a_results.md` |
| Pre-registration (incl. Pareto bars) | `docs/phase1a_preregistration.md` |
| Official config (MNIST) | `configs/phase1a_fixed.yaml` |
| Fashion config (calibrated r*) | `configs/phase1a_fashion.yaml` |
| 20-ep holds | `results/phase1a_diagnostics/gap_imprint_survival_20ep_hold{,_sliding}/` |
| Cohort erosion analysis | `.../gap_imprint_20ep_fixed_vs_sliding_cohort/` |
| MNIST Pareto | `results/phase1a_diagnostics/pareto_comparison/` |
| Fashion calibration | `results/phase1a_diagnostics/fashion_mnist_calibration/` |
| Fashion-MNIST Pareto | `results/phase1a_diagnostics/fashion_mnist_pareto/` (dense scout; plastic under `plastic_calibrated/`) |
| Structure viz example | `results/phase1a_diagnostics/visualizations/` |

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Requires **Python ≥ 3.10**, PyTorch ≥ 2.1, CUDA optional (CPU works; full GT is slow).

```bash
pytest
```

---

## Quick start

### Phase 0

```bash
# CI-scale smoke
python scripts/run_phase0.py --config configs/default.yaml --smoke --condition stationary

# Full multi-seed Phase 0 (new results root)
python scripts/run_phase0.py --config configs/default.yaml --condition both \
  --seeds 0,1,2 --results-root results/phase0_runs/new_primary
```

### Phase 1a

```bash
# Smoke (short budget)
python scripts/run_phase1a.py --config configs/phase1a_fixed.yaml --smoke --condition stationary

# Official locked stack (stationary)
python scripts/run_phase1a.py --config configs/phase1a_fixed.yaml --condition stationary \
  --seeds 0,1,2 --results-root results/phase1a_fixed

# Structure snapshots (opt-in) + viz
# set experiment.record_structure_snapshots: true in config, then:
python scripts/visualize_structure.py \
  --results results/<run>/stationary/seed_0 --animate -o structure.gif

# 20-ep establish + stays hold (+ optional structure viz)
python scripts/run_20ep_hold.py --threshold fixed --seeds 0,1,2 --epochs 20
python scripts/run_20ep_hold.py --threshold sliding --seeds 0,1,2 --epochs 20 --no-viz

# MNIST Pareto (dense grid + score; --skip-train if grid already on disk)
python scripts/run_pareto_comparison.py

# Fashion-MNIST: re-probe r*_ℓ / straddle / growth-fire (required before plastic claims)
python scripts/run_fashion_mnist_calibration.py

# Fashion-MNIST: wide dense scout + plastic under Fashion-calibrated r*_ℓ
python scripts/run_fashion_mnist_pareto.py

# Specialization survival (6-ep primary protocol)
python scripts/run_gap_imprint_survival.py --threshold fixed --seeds 0,1,2
```

Do **not** overwrite historical primary trees in place. New primaries need a **new**
results root after signal, GT, or product-stack changes.

---

## Phase 0 (summary)

**Phase 0** asks whether a locally computable quantity ranks neurons and connections
well enough to stand in for gradient- and ablation-based ground truth — on
stationary MNIST and after task shifts — *before* growth/pruning.

| Gate | Result |
|------|--------|
| Primary B1 (label permutation) | **`PASS`**, `phase1_go=true`, `best_local=activation` |
| Contingency B3 (pixel permutation) | **`PASS`**, `phase1_go=true` |
| Recommended local signal | **`activation`** |
| GEVB-RF as gradient proxy | **not validated** under this design |

**Documents:**  
[`docs/phase0_results.md`](docs/phase0_results.md) ·  
[`docs/phase0_consolidated_results.md`](docs/phase0_consolidated_results.md) ·  
[`docs/design/phase0_local_importance_signal_validation.md`](docs/design/phase0_local_importance_signal_validation.md)

Frozen B1/B3 trees (pre training-RNG-isolation patch):  
`results/historical/pre_patch_rng_isolation/`

### Phase 0 headline findings

1. **`activation`** is a robust current-task edge importance proxy vs `grad_abs`.
2. **`eligibility_residual_energy`** is useful for shift/lag dynamics.
3. **`eligibility_gevb_rf`** is not gradient-aligned under backprop-trained MLPs here.
4. Infrastructure: probe cache + training-RNG isolation around GT.

---

## Package layout

```
configs/                 # Phase 0 + Phase 1a YAMLs
src/neuronca/
  models/                # MLP + DynamicMLP (masked capacity, coords, birth provenance)
  data/                  # MNIST / Fashion-MNIST, TaskContext, B1/B2/B3 transforms
  training/              # loops, plastic_loop, isolation
  plasticity/            # rules, orchestrator, thresholds, structure snapshots
  signals/               # local importance candidates
  importance/            # grad_abs, ablation, fisher, probe cache
  experiments/           # stationary + task_shift runners
  eval/                  # correlations, lag recovery, verdict
  io/                    # config, ledger, artifacts, seed
scripts/
  run_phase0.py
  run_phase1a.py
  run_gap_imprint_survival.py
  run_20ep_hold.py
  run_pareto_comparison.py
  run_fashion_mnist_pareto.py
  run_structure_viz_example.py
  visualize_structure.py
  analyze_*.py
docs/
  phase1a_results.md
  phase1a_preregistration.md
  phase0_*.md
  design/
tests/
results/
  phase1a_diagnostics/   # holds, Pareto, batteries, viz
  historical/
```

### Notable configs

| Config | Use |
|--------|-----|
| `configs/default.yaml` | Full Phase 0 |
| `configs/phase1a_fixed.yaml` | **Official Phase 1a full stack** |
| `configs/phase1a_sliding.yaml` | Sliding threshold (held / long-horizon B) |
| `configs/phase1a_baseline_fixed_topo.yaml` | Dense fixed-topology baseline cells |
| `configs/stationary.yaml` / `task_shift.yaml` | Phase 0 conditions |

---

## Important engineering contracts

| Contract | Where |
|----------|--------|
| Signals **disabled** during GT (no EMA updates) | `InstrumentedTrainer` / `signals_disabled` |
| Training RNG save/restore around GT | `run_gt(isolate_training_rng=True)` |
| Plasticity: local signals only (no labels/grads in rules) | `plasticity/orchestrator.py` |
| Structure snapshots opt-in (default off) | `experiment.record_structure_snapshots` |
| Ledger freezes config content hash + thresholds | `io/ledger.py` |

Changing signal, GT, or product-stack code after a freeze → **new experiment id**,
new results root.

---

## What’s next

Phase 1a mechanism and MNIST efficiency comparisons; Fashion requires dataset-calibrated r*
are **closed** for the locked stack. Natural directions (not all in-tree yet):

- Harder tasks / recipes where dense capacity still has **slope** at plastic operating P
- Task-shift continual metrics under the locked stack
- Phase 1b relay / 1c scaffolding only with establishment and efficiency lessons in mind
- Optional longer-horizon (40–60 ep) specialist equilibrium check — named follow-up, not a MNIST re-Pareto

---

## License

Research prototype; no license asserted yet.
