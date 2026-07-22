# Phase 1a diagnostics — index

**Project status: closed.** Map of pre-registered batteries, holds, and chapter
closes under `results/phase1a_diagnostics/` (local trees; results are gitignored
by default — re-run scripts recreate them).

Primary narrative: [`phase1a_results.md`](phase1a_results.md)

## Chapter closes (read these first)

| Chapter | Path under `results/phase1a_diagnostics/` | Headline |
|---------|-------------------------------------------|----------|
| Fashion thrash | `fashion_mnist_calibration/FASHION_CHAPTER_CLOSE.md` | `RATE_CAP_RESOLVES_THRASH` |
| Self-cal r\* | `self_calibrate/SHIP_DECISION.md` | ship **r\* only** |
| Adaptation cost + retention | `adaptation_cost/RESULTS.md` | thesis **not supported** |
| Split→restore | `adaptation_cost/split_restore/RESULTS.md` | `RETENTION_FIXED_WINS` |
| Spatial coherence | `specialization_spatial_coherence/RESULTS.md` | `MIXED_LAYER_DEPENDENT` |
| Feature neighbors Exp 1 | `feature_neighbors_ab/RESULTS.md` | `FEATURE_NEIGHBORS_WORSE` |

## Mechanism & efficiency

| Artifact | Path |
|----------|------|
| 6-ep establish/stays | `gap_imprint_survival_{fixed,sliding}/`, `gap_imprint_survival_comparison.json` |
| 20-ep hold | `gap_imprint_survival_20ep_hold/`, `..._sliding/` |
| Early-cohort erosion | `gap_imprint_20ep_fixed_vs_sliding_cohort/` |
| MNIST Pareto | `pareto_comparison/` |
| Fashion dense scout | `fashion_mnist_pareto/` |

## Growth / birth chain (historical)

| Topic | Path |
|-------|------|
| Target-rate probes | `target_rate_probe.json`, `fashion_mnist_calibration/target_rate_probe.json` |
| τ / thrash (kaiming era) | `tau_sweep*/`, `tau_thrash_detail/`, `post_grace/` |
| Newborn cold-start | `newborn_*`, `birth_bias_alignment/`, `survivor_age_census/` |
| Parent-seed / mistarget | `parent_seed_*`, `trigger_mistarget/` |
| Gated novelty | `novelty_*`, `gated_novelty_battery*/` |
| Gap imprint ops | `gap_imprint_*` |
| Structure viz | `visualizations/` |

## Pre-registrations (in `docs/`)

| Doc | Topic |
|-----|-------|
| `phase1a_preregistration.md` | Mechanism + Pareto |
| `adaptation_cost_preregistration.md` | Mild adaptation cost |
| `adaptation_cost_split_restore_prereg.md` | Retention cell |
| `feature_neighbors_preregistration.md` | Neighbor metric A/B |

## Re-run policy

New runs after signal / GT / product-stack changes need a **new** results root.
Do not overwrite frozen primary trees in place.
