# Phase 0 Consolidated Results

**Purpose of this document:** single reference for what Phase 0 established,
so Phase 1 design can proceed without replaying chat history or raw JSON.
Frozen primary archives live under
`results/historical/pre_patch_rng_isolation/`. Post-patch diagnostics and
follow-ups live under `results/phase0_*` and `results/phase0_diagnostics/`.

---

## 1. Scope and method

Phase 0 validates **local importance signals** as proxies for ground-truth
importance on a fixed-topology MLP (784→256→128→64→10 unless noted), MNIST,
with four candidates:

1. `activation`
2. `hebbian_corr`
3. `eligibility_residual_energy`
4. `eligibility_gevb_rf` (fixed random feedback `B_ℓ`)

Primary GTs: `grad_abs`, `ablation_relu` (`max(0, ΔL)`). Optional: Fisher
(diagnostic only).

Conditions: **stationary** (Task A) and **task_shift** (A→B) with dense lag.
Pre-registered thresholds (Spearman floors, recovery, weak_shift, GEVB FA angles)
were frozen before primary multi-seed runs and are **not** retuned after results.

**Not in Phase 0:** NCA, orchestrator, growth/pruning, architecture search.

---

## 2. Infrastructure fixes (correctness of later results)

### 2.1 Probe-cache speedup

Full-GT snapshots were ~99% DataLoader/transform cost. Snapshot-scoped probe
tensor cache cuts full snapshot wall from ~minutes to ~3s without changing GT
math on fixed weights (`torch.equal` loader vs cache).

### 2.2 Training-RNG isolation

**Bug (pre-patch):** each probe `DataLoader` iterator advanced the **global CPU
RNG** (`_BaseDataLoaderIter._base_seed`). Uncached ablation did this once per
unit → post-snapshot train shuffle order depended on ablation unit count /
`edge_ablation_sample`.

**Fix:** `run_gt` always saves/restores `torch.get_rng_state()` (and CUDA RNG)
and asserts equality. Same posture as signal-trace isolation.

**Implication for archives:** B1/B3 primaries are **pre-patch**. They remain
valid for fixed configs; do not bit-reproduce them with different edge sample
counts or post-patch code. See
`results/historical/pre_patch_rng_isolation/README.md`.

Controlled post-patch checks (each with an on-disk artifact):

| Check | Result | Artifact |
|-------|--------|----------|
| Train-only A/A (no GT, 3 epochs) | bit-identical | `results/phase0_profile/train_only_aa_repro.json` |
| Snapshots interleaved A/A (same path, empty signals, workers=0) | bit-identical | `results/phase0_profile/snapshots_interleaved_aa_repro.json` |
| Pre-cache vs cache minimal harness + RNG isolation | bit-identical | `results/phase0_profile/precache_vs_cache_rngiso_repro.json` |
| **Full realism A/A** (num_workers=**2**, all **4** live signals, full GT, 3 epochs × 2) | **bit-identical** (loss, weights, signal scores, GT fingerprints; RNG restored after each snap) | **`results/phase0_profile/full_realism_aa_repro.json`** |

The full-realism row is the production-config confirmation (not merely the
minimal harness). Do not treat unchecked production realism as proven if that
JSON is missing.

---

## 3. Activation robustness (stationary ranking)

### 3.1 Main finding

Across B1/B3 historical stationaries, post-patch 20-epoch stationary, truncated
6-epoch stationary, and digit-split `late_a` (fixed 5-way head, see §4.3):

- **`activation` is best_local** for edge × `grad_abs` on essentially every seed.
- Mean edge ρ(activation, grad_abs) ≈ **0.73** (20ep) / **0.76** (6ep residual),
  clearing the 0.60 stationary bar.
- `eligibility_gevb_rf` is consistently last (~0.27–0.32).
- Mid-tier: residual energy ≈ hebbian; order of 2nd/3rd can flip seed-to-seed
  when those ρ differ by ≲0.01.

### 3.2 Convergence (truncated epochs) — controlled framing

Single-variable post-patch **6 vs 20 epochs**:

- **best_local** stays activation; **mean** rank order unchanged.
- Per-seed full order is **not** fully stable in the mid-tier (hebbian vs residual).
- Do **not** cite “order identical on every seed vs pre-patch 20ep” as a clean
  epochs-only result (that comparison mixed code path + epochs).

### 3.3 Neuron-level rankings (existing CSVs)

For **edge × grad_abs**, neuron overall order **agrees** with edge order
(activation first, gevb last) on B1/B3/truncated stationaries.

For **ablation_relu**, **order disagrees**: neuron-level ranks
`eligibility_residual_energy` slightly above `activation` (both ~0.38), while
edge-level ranks activation first but all ablation ρ are weak (~0.15–0.20).
**Granularity-dependent:** ablation-based ranking is not interchangeable between
edge and neuron aggregates. Flag, do not quietly average away.

Detail: `results/phase0_diagnostics/neuron_vs_edge_rankings.md`.

---

## 4. Task-shift and recovery

### 4.1 B1 / B3 (historical pre-patch)

Both reassignment shifts (labels / pixels). Activation wins stationary; lag
recovery under first-crossing `t_star_abs` is messy for activation.

### 4.2 Frozen-weight lag artifact and `t_star_sustained`

**Problem:** `t_star_abs` (first ρ ≥ 0.60) marks activation “recovered” at
**t=0** on B1 all seeds, then the curve collapses before genuine recovery.

**Diagnostic (not a new gate):** `t_star_sustained` = first lag time after which
ρ stays ≥ 0.60 for **all** subsequent points through t=500.

Offline B1/B3 comparison (`results/phase0_diagnostics/sustained_recovery_report.md`):

| Exp | Candidate | Pattern |
|-----|-----------|---------|
| B1 | activation (all seeds) | abs=0, sustained=100 |
| B1 | residual_energy seed2 | abs=0, sustained=50 |
| B3 | activation | abs=sustained (100–125); no t=0 artifact |
| B1/B3 | hebbian | abs≈sustained (late) |
| B1/B3 | gevb | never recovers |

Runner now writes both metrics to `time_lag.csv` / `lag_recovery` for new runs.
**`recovered_abs` / verdict gates still use only `t_star_abs`.**

### 4.3 Digit-split (B2) — new post-patch run

**What was actually tested:** fixed **5-way** head throughout. Phase A = digits
0–4; phase B = digits 5–9 remapped to 0–4. The network never expands its output
capacity and never has to represent more than five classes at once. This is a
**hard distributional / class-set replacement** shift (new digit range, same
task shape) — closer in spirit to an extreme B3 than to orchestrator-style
growth. It is **not** genuine expanding-head class addition or class
obsolescence that requires more representational capacity than the model started
with.

Root: `results/phase0_digit_split/`.

- late_a ranking: **activation still wins** every seed (same as B1/B3).
- mean gt_shift_disagreement ≈ 0.56 (not weak_shift).
- Recovery (sustained): residual energy clean (50–75); activation abs=0 but
  sustained by 50–75; hebbian late; gevb fails.
- Aggregate CLI FAIL only because stationary was not co-run in that root.

**Conclusion (scoped):** under this fixed-head digit-range replacement, activation
remains the ranking winner; residual energy remains the cleanest lag recovery.
**Not tested:** growth under an expanding output head when new classes require
additional capacity (see §8).

---

## 5. GEVB-RF / feedback-alignment diagnostic

### 5.1 Corrected claim

The reasoning net is **backprop-trained**. `B_ℓ` only builds the GEVB-RF
importance signal. This project never trained weights via FA updates.

**Accurate claim:** GEVB-RF’s fixed-projection signal never becomes
gradient-aligned (**FA angle stays ~90°**) under a backprop-trained MLP on this
task, across LR ∈ {0.001, 0.01, 0.05}, `B_ℓ` scale ∈ {0.1×, 1×, 10×}, and a
40-epoch arm at defaults — **not** “feedback alignment as a training method
fails at this scale.”

### 5.2 Sweep outcome

`results/phase0_gevb_fa_sweep/`: all cells **STALLED**, all train successfully
(acc ≥ 0.90). `B_ℓ` scale is inert for angle/Spearman (scale invariance).
Doubling epochs does not help. GEVB ρ stays ~0.19–0.33 (below 0.60).

**Phase 0 stance:** do not promote GEVB-RF as a validated local importance
candidate for this design without a different signal definition or training
regime outside Phase 0 scope.

---

## 6. Fisher ground truth (diagnostic)

Stationary post-patch with `gt.fisher: true`
(`results/phase0_fisher_stationary/`). No Fisher pass/fail threshold.

Mean late **edge** Spearman:

| Candidate | grad_abs | fisher | ablation_relu |
|-----------|---------:|-------:|--------------:|
| activation | 0.732 | 0.675 | 0.199 |
| hebbian_corr | 0.536 | 0.504 | 0.145 |
| residual_energy | 0.542 | 0.474 | 0.189 |
| gevb_rf | 0.289 | 0.300 | 0.099 |

Activation remains strongest for **retention-oriented** Fisher as well as
current-loss `grad_abs`. Ablation overall correlations stay weak for all locals
at edge level.

---

## 7. What Phase 0 establishes for Phase 1

1. **Activation magnitude** is a robust, cheap local signal for current-task
   edge importance vs `grad_abs` (and strong vs Fisher) under MNIST MLP
   stationarity and several shift types (B1 label reassignment, B3 pixel
   permutation, B2 fixed-head digit-range replacement). Expanding-head class
   addition remains untested.
2. **Residual-energy eligibility** is competitive mid-tier and often the cleanest
   **lag recovery** under sustained metrics.
3. **GEVB-RF** as implemented is not a usable gradient proxy here (FA stalled;
   low ρ) under backprop training.
4. **Ablation_relu** is a hard GT; edge-level local correlations are weak —
   do not over-interpret small edge ablation ρ rankings. Neuron ablation
   rankings can reorder top candidates.
5. **Infrastructure:** probe cache + RNG isolation are required for correct
   multi-epoch GT-interleaved training; historical primaries are labeled
   pre-patch.

---

## 8. Still open (not closed by Phase 0)

| Item | Status |
|------|--------|
| Harder data (e.g. Fashion-MNIST) | not run |
| Reduced capacity (smaller MLP) | not run |
| **Expanding-output-head class addition** (more classes / head growth at B; the condition a growth orchestrator must react to) | **not run** — B2 digit-split used a fixed 5-way head only |
| Sustained recovery as a **gate** (replace `t_star_abs`) | deliberate non-goal; diagnostic only |
| Fisher as a pass/fail threshold | not pre-registered |
| Digit-split co-run with full stationary multi-seed aggregate | task_shift-only root; ranking at late_a reported |
| Harmful-unit detection (`max(0,ΔL)` scoping) | known limitation of ablation ranking transform |
| FA under weight updates driven by FA signal | out of design (backprop reasoning net) |

---

## 9. Artifact map

| Location | Content |
|----------|---------|
| `results/historical/pre_patch_rng_isolation/` | Frozen B1/B3 primaries + README |
| `results/phase0_truncated_epochs/` | 6ep stationary + controlled ranking vs post-patch 20ep |
| `results/phase0_stationary_20ep_postpatch/` | Post-patch 20ep stationary baseline |
| `results/phase0_gevb_fa_sweep/` | FA LR×scale grid + 40ep arm |
| `results/phase0_digit_split/` | B2 digit-split task_shift |
| `results/phase0_fisher_stationary/` | Fisher GT stationary |
| `results/phase0_diagnostics/` | Sustained recovery + neuron ranking analyses |
| `scripts/analyze_sustained_recovery.py` | Offline t*_sustained |
| `scripts/analyze_neuron_rankings.py` | Offline neuron vs edge |

---

## 10. Phase 1 pointer

Default local signal for growth/pruning experiments should treat **activation**
as the primary current-task proxy, monitor **residual-energy eligibility** for
shift recovery dynamics, and **not** rely on GEVB-RF without redesign.
Use post-patch trainers only. Prefer **sustained** lag diagnostics when
interpreting recovery curves, even if gates still cite first-crossing for
historical continuity.
