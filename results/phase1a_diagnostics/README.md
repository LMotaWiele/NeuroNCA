# Phase 1a diagnostics — index

**Project status: closed.** This directory holds pre-registered batteries,
holds, and chapter closes for Phase 1a and post-mechanism follow-ups.

Primary narrative write-up: [`docs/phase1a_results.md`](../../docs/phase1a_results.md)

## Chapter closes (read these first)

| Chapter | Path | Headline |
|---------|------|----------|
| Fashion thrash | `fashion_mnist_calibration/FASHION_CHAPTER_CLOSE.md` | `RATE_CAP_RESOLVES_THRASH` |
| Self-cal r\* | `self_calibrate/SHIP_DECISION.md` | ship **r\* only** |
| Adaptation cost + retention | `adaptation_cost/RESULTS.md` | thesis **not supported** |
| Split→restore | `adaptation_cost/split_restore/RESULTS.md` | `RETENTION_FIXED_WINS` |
| Spatial coherence | `specialization_spatial_coherence/RESULTS.md` | `MIXED_LAYER_DEPENDENT` — no regional build |
| Feature neighbors Exp 1 | `feature_neighbors_ab/RESULTS.md` | `FEATURE_NEIGHBORS_WORSE` — no Exp 2 |

## Mechanism & efficiency

| Artifact | Path |
|----------|------|
| 6-ep establish/stays (fixed / sliding) | `gap_imprint_survival_{fixed,sliding}/`, `gap_imprint_survival_comparison.json` |
| 20-ep hold fixed / sliding | `gap_imprint_survival_20ep_hold/`, `..._sliding/` |
| Early-cohort erosion | `gap_imprint_20ep_fixed_vs_sliding_cohort/` |
| MNIST Pareto | `pareto_comparison/` |
| Fashion dense scout (+ invalid plastic) | `fashion_mnist_pareto/` |

## Growth / birth chain (historical diagnostics)

| Topic | Path |
|-------|------|
| Target-rate probes | `target_rate_probe.json`, `fashion_mnist_calibration/target_rate_probe.json` |
| Structure liveness | `structure_liveness.json` |
| τ / thrash (kaiming era) | `tau_sweep/`, `tau_sweep_stability/`, `tau_thrash_detail/`, `post_grace/` |
| Newborn cold-start chain | `newborn_*`, `birth_bias_alignment/`, `survivor_age_census/` |
| Parent-seed / mistarget | `parent_seed_*`, `trigger_mistarget/`, `prune_ablation_mismatch/` |
| Gated novelty | `novelty_*`, `gated_novelty_battery*/`, `growth_trigger_candidates/` |
| Gap imprint ops | `gap_imprint_*` (bias, preact, birth gate, operating point) |
| Fullstack smoke | `fullstack_combined_check/` |
| Structure viz example | `visualizations/` |

## Fashion thrash chain detail

| Piece | Path |
|-------|------|
| Calibration root | `fashion_mnist_calibration/` |
| Fixed vs plastic volatility | `.../FIXED_VS_PLASTIC_VOLATILITY.md` |
| Grow/prune 2×2 | `.../grow_prune_2x2/` |
| Growth locus / worsened-vs-crash | `.../growth_locus_validation/` |
| Rate-cap ablate | `.../rate_cap_ablate/` |

## Adaptation & neighbor experiments

| Piece | Path |
|-------|------|
| Mild digit-split / B1 | `adaptation_cost/digit_split/`, `.../b1_label_perm/` |
| Split→restore | `adaptation_cost/split_restore/` |
| Spatial re-slice | `specialization_spatial_coherence/` |
| Feature vs grid neighbors | `feature_neighbors_ab/{grid,feature}/` |

## Note on git

`results/**` is gitignored by default (see repo `.gitignore`). Keep local
trees for re-analysis; re-runs should use a **new** results root after code
changes that affect signal, GT, or the product stack.
