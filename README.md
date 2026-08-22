# NeuroNCA

Research codebase for **structural plasticity** on small MLPs. The reasoning net
always trains with standard **backpropagation**. Local activation signals are
observed alongside training and, in Phase 1a, drive **homeostatic grow/prune** of
hidden neurons and adjacent edges (never a topology optimizer over multi-hop
graphs).

The in-scope chapters in this repository are **closed**. Findings below are the
bounded, publishable conclusions — not an open research roadmap.

Package: `src/neuronca/` · layout rules: `AGENTS.md`.

## Project status: **CLOSED**

| Chapter | Status | Headline |
|---------|--------|----------|
| **Phase 0** — local importance signals | **Closed** | `activation` best current-task proxy; GEVB-RF not validated |
| **Phase 1a** — mechanism (establish / stays specialized) | **Closed** | Gap-imprinted specialists establish and stay specialized (~91% @ 6 ep; ~79% @ 20 ep) |
| **Phase 1a** — MNIST efficiency (Pareto vs dense) | **Closed** | `PARETO_BOTH_FAIL` — dense already flat; plastic not competitive |
| **Fashion thrash** | **Closed** | Need signal real; thrash = insertion volume → `RATE_CAP_RESOLVES_THRASH` (cap=1) |
| **Self-calibrating r\*_ℓ** | **Closed** | Ship **r\* only**; prune/straddle self-cal not shipped |
| **Adaptation cost** (mild shifts) | **Closed** | `ADAPT_COST_FIXED_WINS` — everyone recovers in 1 ep; plastic +~35% wall-clock |
| **Adaptation retention** (split→restore) | **Closed** | `RETENTION_FIXED_WINS` — fixed retains more of A after B; thesis not supported |
| **Spatial specialization coherence** | **Closed** | `MIXED_LAYER_DEPENDENT` — no full regional-consolidation build |
| **Feature-space neighbors** (Exp 1) | **Closed** | `FEATURE_NEIGHBORS_WORSE` — no Exp 2 (cohort consolidation) |

**Primary write-ups**

- Phase 1a: [`docs/phase1a_results.md`](docs/phase1a_results.md) · [`docs/phase1a_preregistration.md`](docs/phase1a_preregistration.md)
- Phase 0: [`docs/phase0_results.md`](docs/phase0_results.md) · [`docs/phase0_consolidated_results.md`](docs/phase0_consolidated_results.md)
- Design: [`docs/design/phase1_design_document.md`](docs/design/phase1_design_document.md) · [`docs/design/invariants.md`](docs/design/invariants.md)

---

## Bounded conclusions (what this project supports)

1. **Local structural plasticity can create specialists.** Under gated novelty +
   gap-imprinted birth, grown units establish and remain specialized on their
   birth gap for a meaningful fraction of lifetime (strong at 6 ep; weaker but
   still above convenience bars at 20 ep). Parent-clone birth does **not** create
   new function.

2. **On stationary MNIST, plasticity is not an efficiency win.** The dense
   accuracy-vs-parameter curve is already flat in the plastic operating region;
   plastic ends near ~96.6% @ ~88k params vs dense ~97.9% @ ~81k
   (`PARETO_BOTH_FAIL`).

3. **Fashion thrash was a rate problem, not a failed need signal.** MNIST r\* does
   not transfer. With Fashion-calibrated / self-cal r\* and
   `max_new_neurons_per_layer=1`, thrash resolves. Trigger tracks error; births
   often help locally.

4. **Self-calibrate r\*_ℓ from a one-epoch probe.** Ship that only. Frozen prune
   z-cut and live per-birth straddle B stay as-is.

5. **Adaptation thesis not supported on digit tasks.** Mild one-way shifts are too
   easy (scratch also recovers in 1 epoch). Under split→restore interference,
   fixed retains **more** of task A than plastic. Stop rule applied — no further
   harder-shift treadmill without a new product mandate.

6. **Regional / feature-neighbor consolidations not justified here.** Grid
   specialization is only partially coherent (h1 yes; stays specialists isolated).
   Replacing grid neighbors with feature-similarity did not clear stationary
   regression bars (`FEATURE_NEIGHBORS_WORSE`).

---

## Official product stack (locked)

`configs/phase1a_fixed.yaml` (MNIST):

| Piece | Setting |
|-------|---------|
| Threshold | **fixed** (sliding held for long-horizon B only) |
| Growth | **gated novelty** (novelty ∧ local CE ∧ \|N\| ≥ **k_min=11**) |
| Birth | **gap_imprinted** (sep-power edges, open-fan scale, straddle bias) |
| Prune | competitive z-score / fraction |
| Grace | 3 (≥ birth-bias steps) |
| r\*_ℓ | probe or self-cal; MNIST defaults (0.77, 1.02, 1.28) |

Fashion product: `configs/phase1a_fashion.yaml` — **self-cal r\*** + **cap=1**.

Neighbor metric defaults to **grid** (`neighbor_metric: grid`). Feature neighbors
remain available as a research flag only; Exp 1 did not promote them.

---

## Setup

Python >= 3.12. PyTorch >= 2.1. CUDA optional (CPU works; full GT is slow).

```bash
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

---

## Reproduce (closed chapters)

Do **not** overwrite frozen result trees in place. New runs need a **new** results
root if you change signal, GT, or product-stack code.

### Phase 0

```bash
python scripts/run_phase0.py --config configs/default.yaml --smoke --condition stationary
python scripts/run_phase0.py --config configs/default.yaml --condition both \
  --seeds 0,1,2 --results-root results/phase0_runs/new_primary
```

### Phase 1a core

```bash
python scripts/run_phase1a.py --config configs/phase1a_fixed.yaml --smoke --condition stationary
python scripts/run_gap_imprint_survival.py --threshold fixed --seeds 0,1,2
python scripts/run_20ep_hold.py --threshold fixed --seeds 0,1,2 --epochs 20
python scripts/run_pareto_comparison.py
```

### Fashion / self-cal / adaptation / neighbor A/B

```bash
python scripts/run_fashion_mnist_calibration.py
python scripts/run_fashion_rate_cap_ablate.py
python scripts/run_self_calibrate_battery.py --stage all --epochs 8

python scripts/run_adaptation_cost.py --seeds 0,1,2
python scripts/run_adaptation_split_restore.py --seeds 0,1,2

python scripts/analyze_specialization_spatial_coherence.py
python scripts/run_feature_neighbors_ab.py --seeds 0,1,2 --epochs 20
```

### Structure visualization

```bash
# set experiment.record_structure_snapshots: true, then:
python scripts/visualize_structure.py \
  --results results/<run>/stationary/seed_0 --animate -o structure.gif
python scripts/run_structure_viz_example.py
```

---

## Artifact map

| Chapter | Results root / doc |
|---------|-------------------|
| Phase 0 primary (historical freeze) | `results/historical/pre_patch_rng_isolation/` |
| Phase 0 write-ups | `docs/phase0_*.md` |
| Phase 1a diagnostics (all batteries) | `results/phase1a_diagnostics/` — see index below |
| Mechanism 6-ep / 20-ep | `gap_imprint_survival_*`, `gap_imprint_survival_20ep_hold{,_sliding}/` |
| MNIST Pareto | `pareto_comparison/` |
| Fashion thrash close | `fashion_mnist_calibration/FASHION_CHAPTER_CLOSE.md` |
| Self-cal ship decision | `self_calibrate/SHIP_DECISION.md` |
| Adaptation cost + retention | `adaptation_cost/RESULTS.md`, `adaptation_cost/split_restore/` |
| Spatial coherence | `specialization_spatial_coherence/RESULTS.md` |
| Feature neighbors Exp 1 | `feature_neighbors_ab/RESULTS.md` |
| Pre-registrations | `docs/*preregistration*.md`, `docs/adaptation_cost_*.md`, `docs/feature_neighbors_preregistration.md` |

Diagnostic index: [`docs/phase1a_diagnostics_index.md`](docs/phase1a_diagnostics_index.md)
(also mirrored under `results/phase1a_diagnostics/README.md` for local trees)

---

## Package

```
configs/                 # Phase 0 + Phase 1a YAMLs (top-level by design)
src/neuronca/
  models/                # MLP + DynamicMLP (masks, coords, birth provenance)
  data/                  # MNIST / Fashion-MNIST / KMNIST, TaskContext, shifts
  training/              # loops, plastic_loop, isolation
  plasticity/            # rules, orchestrator, neighbors, self_cal, snapshots
  signals/               # local importance candidates
  importance/            # grad_abs, ablation, fisher, probe cache
  experiments/           # stationary + task_shift runners
  eval/                  # correlations, lag recovery, verdict
  io/                    # config, ledger, artifacts, seed
scripts/                 # campaign runners + analyzers
docs/                    # preregs, results write-ups, design
tests/                   # flat test_<topic>.py
data/                    # dataset downloads
results/                 # experiment trees (do not rename)
```

Layout rationale: [`docs/design/layout.md`](docs/design/layout.md).
Snapshot / ledger / RNG contracts: [`docs/design/invariants.md`](docs/design/invariants.md).

### Scripts (by chapter)

| Script | Chapter |
|--------|---------|
| `run_phase0.py` | Phase 0 |
| `run_phase1a.py` | Phase 1a harness |
| `run_gap_imprint_survival.py` | Mechanism primary (6 ep) |
| `run_20ep_hold.py` | Mechanism 20-ep hold |
| `run_pareto_comparison.py` | MNIST Pareto |
| `run_fashion_mnist_calibration.py` | Fashion r\* + thrash prechecks |
| `run_fashion_rate_cap_ablate.py` | Fashion rate-cap close |
| `run_fashion_*_ablate.py` / `growth_locus_*` | Fashion thrash chain |
| `run_self_calibrate_battery.py` | Self-cal §4.1–4.6 |
| `run_adaptation_cost.py` | Mild adaptation cost |
| `run_adaptation_split_restore.py` | Retention under interference |
| `analyze_specialization_spatial_coherence.py` | Spatial go/no-go |
| `run_feature_neighbors_ab.py` | Feature neighbors Exp 1 |
| `visualize_structure.py` / `run_structure_viz_example.py` | Structure viz |

### Notable configs

| Config | Use |
|--------|-----|
| `configs/default.yaml` | Full Phase 0 |
| `configs/phase1a_fixed.yaml` | **Official Phase 1a stack (MNIST)** |
| `configs/phase1a_sliding.yaml` | Sliding threshold (held / long-horizon) |
| `configs/phase1a_fashion.yaml` | Fashion product (self-cal r\* + cap=1) |
| `configs/phase1a_baseline_fixed_topo.yaml` | Dense fixed-topology cells |
| `configs/stationary.yaml` / `task_shift*.yaml` | Phase 0 conditions |

---

## What this repo is *not* claiming

- Plasticity as a general continual-learning win on these digit tasks
- Competitive stationary efficiency vs dense MLPs on MNIST at the locked size
- That grid coordinates form reliable specialist “regions” for consolidation
- That feature-space neighbors improve the locked stack without regression
- Phase 1b relay / 1c scaffolding (never opened)

## Docs

Write-ups and preregs live under [`docs/`](docs/). Design and invariants under
[`docs/design/`](docs/design/). Optional figure packaging is out of scope here.

---

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
