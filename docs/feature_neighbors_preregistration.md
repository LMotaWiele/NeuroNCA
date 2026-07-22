# Pre-Registration: Feature-Space Neighborhoods (Experiment 1)

## Status

**Locked before run.** First test of the neighbor-definition substitution.
Single-variable A/B: replace the orchestrator's neighbor-lookup (currently
grid-distance within radius R) with feature-similarity, changing NOTHING else.
Judged on whether the same local rules produce better structure over a
functionally-meaningful neighborhood than over a spatially-random one. Gates
whether Experiment 2 (feature-cohort consolidation for retention) is worth
building.

**Runner:** `scripts/run_feature_neighbors_ab.py`  
**Config flag:** `PlasticityConfig.neighbor_metric ∈ {grid, feature}`  
**Results:** `results/phase1a_diagnostics/feature_neighbors_ab/`

Context: grid coordinates were shown (spatial-coherence check) to carry little
functional information for long-lived specialists — a neuron's grid-neighbors
are often a random functional assortment. Hypothesis: redefining "neighbor" as
feature-similar makes every existing local rule (novelty, competitive pruning)
operate over a meaningful neighborhood, without adding any new rule.

## 1. Hypothesis (pre-registered, directional)

> With neighbor = feature-similarity instead of grid-distance, and all other
> rules unchanged, competitive pruning removes genuinely more redundant units
> (lower own-subset ablation impact among pruned units) than under grid-neighbors,
> at comparable-or-better stationary accuracy and stability.

## 2. The Single Variable

**Change exactly one function: `get_neighbors(unit)` / neighbor graph.**

| Arm | Neighbor definition |
|-----|---------------------|
| **grid** (control) | Same-layer units within grid radius R |
| **feature** (treatment) | Same-layer top-k by probe-activation cosine similarity |

Scope of the graph (both arms):

- Same-layer only (same candidate pool / adjacency scope as grid novelty).
- Used by: **gated novelty** (nbr_max / density floor) and **competitive neuron
  z-score** (z among `N(i) ∪ {i}`).
- **Not** changed: edge incentive (cross-population placement still uses grid R),
  gap-imprint seed-edge radius, coords storage, birth jitter, all constants.

Neighborhood-local neuron z applies to **both** arms so the comparison is
metric-only (feature vs grid), not "local feature vs historical global z."

## 3. Feature Signature Definition (pre-registered)

- **Signature:** `sig(i) = [activation_i(x) for x in probe]` on the fixed probe
  set already used for gated novelty / ablation.
- **Similarity:** cosine (primary). Correlation = center-then-cosine fallback;
  record which is used (default cosine).
- **Neighborhood size:** top-k with k = round(mean grid-neighbor count under R)
  on the **current** active population each refresh (matched size; isolates
  metric). Record matched k per layer per epoch.
- **Not** used for the decision: gap-target-subset overlap (secondary log only
  if added later).

## 4. Non-Stationarity Handling (pre-registered)

- **Refresh cadence N:** once per plasticity step (= once per epoch). Do not
  recompute every SGD step. Record N.
- **Neighbor-churn guard:** per epoch, mean fraction of previous neighbors lost
  (`|N_prev \\ N_new| / |N_prev|` over units with |N_prev|>0). Bound = **0.5**.
  If mean churn > 0.5 → flag `NEIGHBORHOOD_UNSTABLE`.
- **Pre-committed response:** enable EMA-smoothed signatures (`mu=0.5`) and
  re-refresh that step. Do not score the decision criteria until post-response
  mean churn is ≤ bound (report raw + post-response churn).

## 5. Decision Criteria (pre-registered, dual)

Both arms, seeds {0,1,2}, stationary MNIST, locked Phase 1a stack, **20 epochs**.

### Primary — redundancy-detection quality

Own-subset ablation ΔL⁺ of **actually pruned** units (preferred-class subset on
probe; Phase-0-style zero incoming+outgoing), measured immediately before prune.

- **Feature wins:** mean pruned-unit own-subset ΔL is ≥ **25% lower** under
  feature than grid, **3/3 seeds** directionally (feature < grid).
- **Feature loses:** feature mean own-subset ΔL higher (worse — cutting more
  useful units). Halt the line.

### Secondary — no stationary regression

- Final test accuracy: feature mean within **1.0 pp** of grid mean (existing
  seed-noise style bar), and no seed with feature worse than grid by > **1.5 pp**.
- Stability: total structure thrash
  (`sum(neurons_grown + neurons_pruned)` over epochs) feature ≤ **1.5×** grid
  seed-mean; max consecutive val-acc swing feature ≤ grid + **2 pp**.

### Verdict

| Label | Condition |
|-------|-----------|
| `FEATURE_NEIGHBORS_BETTER` | Primary win AND secondary no-regression → proceed to Exp 2 |
| `FEATURE_NEIGHBORS_NEUTRAL` | Primary tie, secondary no-regression → proceed to Exp 2, note neutral |
| `FEATURE_NEIGHBORS_WORSE` | Primary loss OR secondary regression → **stop** |

Primary **tie:** neither ≥25% lower with 3/3 directional, nor feature higher
with 3/3 directional against.

## 6. Deliverables

- Config flag `neighbor_metric: grid | feature`
- Per-epoch neighbor-churn log
- Pruned-unit own-subset ablation ΔL, both arms, 3 seeds
- Stationary accuracy + stability, both arms, 3 seeds
- Verdict per §5 + churn-stability result always reported

## 7. Explicit Non-Goals

- No consolidation / retention (Experiment 2)
- No attention / transformer work
- No change to any rule other than neighbor definition
- No dropping of grid coordinates
- No post-hoc constant tuning

## 8. Locked Constants

| Item | Value |
|------|------:|
| Signature | probe activation cosine |
| Matched k | round(mean grid \|N\| under R), per layer, per refresh |
| Refresh N | 1 epoch |
| Churn bound | 0.5 |
| Redundancy margin | ≥25% lower own-subset ΔL, 3/3 directional |
| Acc regression | 1.0 pp mean / 1.5 pp any seed |
| Thrash | ≤1.5× grid |
| Stability response | EMA μ=0.5 on signatures |
