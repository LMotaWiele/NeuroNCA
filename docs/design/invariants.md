# Engineering invariants

One-line constraints that an editor would otherwise break. Code comments point here; they do not restate the full rationale.

Changing signal, GT, or product-stack code after a freeze requires a **new experiment id** and a **new results root**. Do not overwrite frozen trees in place.

## Snapshot isolation (signals off during GT)

GT (gradient saliency, ablation, Fisher) must not call `LocalSignal.update` and must not advance signal EMA buffers. `InstrumentedTrainer.signals_disabled()` is the safety mechanism; bitwise trace equality is the live check (`IsolationGuard`, default `always`).

Rationale: probe forwards would otherwise leak into local scores and confound Spearman vs GT. Isolation is cheap relative to a full Phase 0 run.

## Training RNG save/restore around GT

`run_gt(isolate_training_rng=True)` captures `torch` CPU (+ CUDA) RNG, runs GT, restores, and asserts equality.

Rationale: probe `DataLoader` iterator creation (and per-unit ablation re-iteration) advances the global CPU RNG. Without the bracket, `edge_ablation_sample` / neuron-unit count couple into post-snapshot train shuffle order. Historical B1/B3 primaries under `results/historical/pre_patch_rng_isolation/` were collected **without** this bracket; they remain valid for their frozen config but are not bit-comparable to post-patch runs.

## Probe cache is snapshot-scoped

Each `run_gt` / `snapshot` materializes a fresh `ProbeCache` and (by default) re-materializes after the suite to assert bit-equality. Never reuse a cache across snapshots.

## Plasticity uses local signals only

The orchestrator may read local activation scores, coords, masks, and the optional broadcast residual packet already attached to the trainer. It must not read task labels or gradients when deciding grow/prune. Reasoning-net weight updates remain ordinary backprop.

## Ledger freeze

At primary launch, `{results_root}/meta/` records:

- `experiment_id` = `UTC_date + "_" + first8(sha256(canonical_yaml(pre-stamp config)))`
- resolved config, content SHA-256 (experiment.id forced to null), thresholds, and a hash of `signals/` + `broadcast/` source

Freeze is idempotent when hashes and the experiment-id suffix match. It **refuses to overwrite** when the config hash, signals/broadcast code hash, or id suffix disagree. Thresholds are re-derived from the passed config (disk thresholds are not trusted). Multi-seed runs share one `meta/` — the content hash must be seed-invariant; per-run seed is applied after freeze.

## Neighbor metric default is grid

`PlasticityConfig.neighbor_metric` defaults to `"grid"`. Feature-space neighbors (`"feature"`) remain a research flag; Experiment 1 did not promote them (`FEATURE_NEIGHBORS_WORSE`). Edge incentive still uses grid radius; only same-layer `get_neighbors` scope swaps.

## Task-switch drain protocol

`TaskContext.set_transform` is an in-process assignment. With `num_workers > 0`, workers hold a forked copy and will not see a mid-lifetime flip. At an A→B boundary: drain loader iterators first, do not keep `persistent_workers=True` across the flip (rebuild loaders), prefer `num_workers=0` for probe/GT at the switch.

## Newborn grace vs birth-bias scaffold

`newborn_grace_steps` must be `>= newborn_birth_bias_steps`. Contraction prune must not judge a unit while the temporary birth-bias scaffold is still nonzero.

## GEVB FA-alignment annotation

`eval/gevb_alignment_verdict.py` annotates whether fixed-random feedback was aligned enough for GEVB-RF Spearman to be interpretable. Thresholds live in `configs/default.yaml` `eval.thresholds` and freeze into `meta/thresholds.json`. The annotation never flips `phase1_go`. Do not retune bars from a primary run; a "maybe more training" follow-up needs a new experiment id.
