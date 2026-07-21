# Phase 0 Results Write-Up

Post-run scientific summary for **Local Importance Signal Validation**.
Fill placeholders after a primary multi-seed run (`scripts/run_phase0.py` / `scripts/reproduce.sh`).
Do **not** redefine signal formulas or thresholds after seeing results; archive under a new experiment id if measurement code changes.

**Canonical narrative / Phase 1 reference (full post-patch diagnostics):**
[`docs/phase0_consolidated_results.md`](phase0_consolidated_results.md).

## Pre-patch archives (B1 / B3 primaries)

The original multi-seed B1 and B3 primary trees are **frozen historical** artifacts under:

`results/historical/pre_patch_rng_isolation/`

| | B1 (primary) | B3 (contingency) |
|--|--------------|------------------|
| **Subdir** | `phase0_B1_label_permutation/` | `phase0_B3_pixel_permutation/` |
| **Experiment id** | `2026-07-20_0a69524f` | `2026-07-21_c0916ec7` |
| **Config content hash (prefix)** | `0a69524f…` (`meta/config.content_sha256.txt`) | `c0916ec7…` |
| **Date (UTC, from experiment id)** | 2026-07-20 | 2026-07-21 |
| **Shift** | `label_permutation` | `pixel_permutation` |
| **Seeds** | `0, 1, 2` | `0, 1, 2` |
| **Conditions** | stationary · task_shift | stationary · task_shift |
| **Device (logged)** | cuda | cuda |

They were collected **before** the training-RNG isolation fix around `run_gt` (global CPU RNG was advanced by probe `DataLoader` iterators inside ablation, coupling `edge_ablation_sample` / unit count into post-snapshot train shuffle). Results remain scientifically valid for the fixed config used, but are **not** bit-reproducible under post-patch code or a different `edge_ablation_sample`. See that folder’s `README.md` and §2.2 of `phase0_consolidated_results.md`.

| Field | Value |
|-------|-------|
| **Results root (original)** | `results/phase0/` (B1) / `results/phase0_b3_pixel_perm/` (B3) — **relocated** to historical paths above |
| **Primary artifacts** | `aggregate/verdict.json`, `meta/thresholds.json`, `meta/config.resolved.yaml` |
| **Git commit** | not frozen in ledger; repo HEAD at this write-up is not a substitute for “commit at primary freeze” |

**Post-patch confirmation runs** (20ep stationary, truncated 6ep, digit-split fixed-head, GEVB sweep, Fisher, full-realism A/A) used patched code and are cross-referenced below where relevant; they do **not** replace the frozen primary verdict trees.

---

## Verdict summary

| Field | Value |
|-------|-------|
| **verdict** | **`PASS`** |
| **phase1_go** | **`true`** |
| **best_local** | **`activation`** |
| **stationary_pass** | **`true`** |
| **recovered** | **`true`** |
| **weak_shift** | **`false`** |
| **catastrophic_fail** | **`false`** |

Key fields from B1 `aggregate/verdict.json`:

```json
{
  "verdict": "PASS",
  "phase1_go": true,
  "best_local": "activation",
  "stationary_pass": true,
  "recovered": true,
  "weak_shift": false,
  "catastrophic_fail": false,
  "mean_rho_by_candidate": {
    "activation": 0.7337971688881654,
    "eligibility_residual_energy": 0.5571110076710685,
    "hebbian_corr": 0.531913239633568,
    "eligibility_gevb_rf": 0.27340560569085487
  },
  "rho_shift_mean": 0.6323269111195896,
  "recovery_rho_mean": 0.7093340386699752,
  "gevb_alignment": {
    "alignment_status": "STALLED",
    "mean_late_fa_deg": 90.36357724209078,
    "mean_early_fa_deg": 89.59448060574677,
    "mean_improvement_deg": -0.7690966363440074
  }
}
```

**B3 contingency companion** (`2026-07-21_c0916ec7`, pixel permutation): same verdict shape — `verdict: PASS`, `phase1_go: true`, `best_local: activation`, `weak_shift: false` (mean `gt_shift_disagreement` ≈ **0.133**, i.e. a genuinely hard shift, not a vacuous one). Stationary Task-A mean ρ table matches B1 (same identity regime). Activation lag recovery under first-crossing is **not** t=0 on B3 (`t*_abs` = 125 / 100 / 125). See §4 of `phase0_consolidated_results.md`.

---

## Pass / partial / fail criteria

Pre-declared thresholds from B1 `meta/thresholds.json` (do not edit post-hoc).

| Criterion | Metric | Threshold (declared) | Observed (B1) | Met? |
|-----------|--------|----------------------|---------------|------|
| **Stationary pass (primary)** | Best local **edge** Spearman vs `grad_abs` at stationary `late`: mean ρ ≥ threshold **and** ≥ 2/3 seeds ≥ seed floor | mean ≥ **0.60**; seed floor **0.50** (min count **2**) | mean=**0.734**; seeds above floor=**3**/3 (0.736 / 0.703 / 0.762) | ☑ |
| **Stationary secondary** | Same best local, **neuron** Spearman vs `ablation_relu` at `late` | mean ρ ≥ **0.50** (informational only) | activation neuron×ablation ≈ **0.382** — below 0.50; **note:** neuron-level ranking vs ablation can put `residual_energy` marginally above `activation`. See `results/phase0_diagnostics/neuron_vs_edge_rankings.md`. | ☑ n/a (informational; not a pass gate) |
| **Shift recovery** | Same `best_local`, edge vs `grad_abs`: finite \(t^*_{\mathrm{abs}}\) ≤ **500** for enough seeds | ≥ **2 of 3** seeds recover | seeds recovered=**3**/3; \(t^*_{\mathrm{abs}}=0\) all seeds under first-crossing rule (**see lag caveat** — frozen-weight artifact; `t_star_sustained` ≈ **100** for genuine hold) | ☑ (per pre-registered rule; see caveat) |
| **Partial pass** | Stationary pass; shift recovery fails | lag is a Phase 1 design constraint | not applicable — recovery passed | ☐ |
| **Fail** | Stationary pass not met | rethink info access | not applicable | ☐ |
| **Catastrophic fail (annotation)** | \(\max_c \bar\rho(c) <\) `fail_rho_ceiling` at stationary late | ceiling **0.30** | max mean ρ=**0.734** (activation) | ☐ (not triggered) |
| **Weak shift** | Mean `gt_shift_disagreement` (ρ_shift) ≥ ceiling | ceiling **0.90** → `PASS_WEAK_SHIFT` if recovered | ρ_shift mean=**0.632** (B1); **0.133** (B3) | ☐ (not triggered) |

### GEVB FA alignment (pre-registered annotation — does not flip `phase1_go`)

Locked in `meta/thresholds.json` **before** the primary run. Source:
`eval/gevb_alignment_verdict.py` + `configs/default.yaml`.

**Metric:** per seed, mean over layers of `fa_signal_angle_deg` at stationary
`late` (from `gevb_alignment.csv`). Aggregate = mean over seeds.

| Status | Rule (declared) | Observed (B1) | Met? |
|--------|-----------------|---------------|------|
| **CONVERGED** | mean_late ≤ **60°** AND ≥ **2/3** seeds have layer-mean ≤ **70°** | mean=**90.36°**; seeds ≤70°=**0**/3 | ☐ (fail) |
| **STALLED** | mean_late ≥ **75°** OR (early→late improvement < **10°** AND late ≥ **70°**) | mean=**90.36°**; Δ early→late=**−0.77°** (worse, not better) | ☑ (triggered) |
| **PARTIAL** | otherwise (with ≥2 seeds) | — | ☐ |
| **INSUFFICIENT_DATA** | <2 seeds with finite late FA (e.g. smoke) | n_seeds=**3** | ☐ |

**`alignment_status = STALLED`**, confirmed on B1, B3, and the dedicated LR × `B_ℓ`-scale diagnostic sweep (3×3 grid + 40-epoch arm, all cells STALLED; `B_ℓ` scale inert for angle/Spearman by scale invariance). See §5 of `phase0_consolidated_results.md` for **corrected scope**: this reflects GEVB-RF’s signal quality **under a backprop-trained reasoning network** — weights are never updated via the FA teaching signal in this design.

**Process rules (also frozen):**
1. Do **not** extend epochs, retune B_ℓ scale, or re-draw feedback based on primary trajectories.
2. “Maybe more training” → **new** `experiment_id`, doubled epochs (40 / 30+30), **same** angle bars only.
3. Alignment status never flips `phase1_go` by itself.

**ρ × alignment matrix (annotation only):**

| alignment | GEVB late edge ρ | Interpretation |
|-----------|------------------|----------------|
| CONVERGED | high / best_local | FA working as theory suggests |
| CONVERGED | low / last | Serious: teaching signal aligned, proxy still poor |
| STALLED | low / last | Expected under FA lag; not validated at this budget |
| STALLED | high | Suspicious; investigate before trusting |
| PARTIAL | anything | Inconclusive for FA |

Observed: **STALLED + last** (mean GEVB edge ρ **0.273**).

### Verdict enum (reference)

```
if not stationary_pass:
    verdict = FAIL; phase1_go = false
elif recovered and not weak_shift:
    verdict = PASS; phase1_go = true
elif recovered and weak_shift:
    verdict = PASS_WEAK_SHIFT; phase1_go = false
else:
    verdict = PARTIAL_PASS; phase1_go = false
```

**Phase 1 go/no-go:** `phase1_go == true` **only** when `verdict == PASS`. `PASS_WEAK_SHIFT` is **not** a Phase 1 green light.

### Per-candidate stationary means (edge vs `grad_abs`, `late`)

B1 primary (Task A identity). B3 stationary archive reports the **same** mean table (shared Task-A regime).

| Candidate | ρ seed 0 | ρ seed 1 | ρ seed 2 | mean ρ | Notes |
|-----------|----------|----------|----------|--------|-------|
| `activation` | 0.7365 | 0.7029 | 0.7620 | **0.7338** | winner; clears floor on all 3 seeds |
| `eligibility_residual_energy` | 0.5354 | 0.5512 | 0.5847 | **0.5571** | mid-tier; cleanest lag-recovery profile under **sustained** metric |
| `hebbian_corr` | 0.5060 | 0.5097 | 0.5800 | **0.5319** | mid-tier; can swap rank with residual_energy seed-to-seed (Δ≲0.01) |
| `eligibility_gevb_rf` | 0.2898 | 0.2445 | 0.2859 | **0.2734** | last on every seed; FA alignment stalled |

`best_local` = **`activation`** (argmax mean ρ). Also robust across B3 pixel-shift, post-patch 20ep / controlled 6ep stationary, Fisher GT, and fixed-head digit-range replacement — with caveats in consolidated §3 / §4.3 / §6.

**Post-patch 20-epoch stationary control** (`results/phase0_stationary_20ep_postpatch/`): activation mean **0.732** (0.733 / 0.703 / 0.760). Same winner.

---

## Per-layer notes

Primary pass/fail uses **connection-level** (edge) Spearman.

**3-seed mean, stationary `late`, activation × `grad_abs` (B1 `correlations.csv`, `is_overall=false` layers):**

| Layer | best_local edge ρ (mean over seeds) | Notes |
|-------|-------------------------------------|-------|
| `fc1` | **0.712** | lowest of four; still ≫ 0.60 |
| `fc2` | **0.816** | |
| `fc3` | **0.895** | strongest layer |
| `fc_out` | **0.758** | |
| overall (z-score concat) | **0.734** | primary gate |

These are aggregated from frozen CSVs (not single-seed early-tag spot checks). Source script pattern: mean over seeds of per-layer edge rows at `snapshot_tag=late`.

Secondary / diagnostic:

| Pair | Layer scope | Observed ρ (mean) | Comment |
|------|-------------|-------------------|---------|
| activation vs `grad_abs` (edge) | overall | **0.734** | primary |
| activation vs `ablation_relu` (edge) | overall | **0.188** | weak |
| activation vs `ablation_relu` (neuron) | overall | **0.382** | secondary bar 0.50 not met (informational) |
| residual_energy vs `grad_abs` (neuron) | overall | **~0.728** | nearly ties activation on neuron×grad_abs |
| activation vs **fisher** (edge, post-patch stationary) | overall | **0.675** | diagnostic only |

**Observations:**

- Overall edge ρ is not carried by a single layer; `fc3` highest, `fc1` lowest; all strong for activation.
- Scalar residual-energy is competitive mid-tier on edge; on **neuron × grad_abs** it nearly matches activation.
- **Neuron × ablation_relu** can rank residual_energy slightly above activation — **granularity-dependent**; do not treat edge and neuron ablation rankings as interchangeable (`results/phase0_diagnostics/neuron_vs_edge_rankings.md`).
- Snapshot isolation / probe-cache integrity / training-RNG restore: green on post-patch controlled A/As, including full realism (workers=2, 4 live signals) — artifact `results/phase0_profile/full_realism_aa_repro.json` (see consolidated §2.2).

---

## Time-lag notes

Task-shift recovery uses the **global** `best_local` (frozen from stationary) vs `grad_abs` on the dense lag window.

### Best-local (`activation`) recovery, B1, edge vs `grad_abs`

| Seed | \(t^*_{\mathrm{abs}}\) (first ρ ≥ 0.60 within 500) | \(\rho^{\mathrm{rec}}\) (first-crossing) | \(\rho_{\mathrm{pre-switch}}\) (`late_a`) | \(\rho_{\mathrm{max}}\) in window | \(\rho\) @ window end | Recovered (gate)? |
|------|-----------------------------------------------------|------------------------------------------|-------------------------------------------|-----------------------------------|----------------------|-------------------|
| 0 | **0.0** | 0.7106 | 0.7246 | 0.9180 | 0.9077 | ☑ |
| 1 | **0.0** | 0.6794 | 0.7125 | 0.8998 | 0.8857 | ☑ |
| 2 | **0.0** | 0.7380 | 0.7675 | 0.8973 | 0.8925 | ☑ |
| **Aggregate** | \(t^*_{\mathrm{abs}}=0\) all seeds | `recovery_rho_mean` ≈ **0.709** | | | | **`recovered` = true** |

**Critical caveat — do not read \(t^*_{\mathrm{abs}}=0\) as instant adaptation:** under label permutation, inputs are unchanged at the switch, so frozen activations can look spuriously aligned with the *new* `grad_abs` at t=0. Offline `t_star_sustained` (first t after which ρ stays ≥ 0.60 for **all** subsequent lag points through 500) puts genuine hold-through recovery at **t = 100** for activation on B1 all three seeds — not t=0. Diagnostic only; **does not** replace `t_star_abs` in the verdict gate. Report: `results/phase0_diagnostics/sustained_recovery_report.md`.

| Seed | \(t^*_{\mathrm{abs}}\) | \(t^*_{\mathrm{sustained}}\) (activation, B1) |
|------|------------------------:|-----------------------------------------------:|
| 0–2 | 0 | **100** |

`eligibility_residual_energy` typically has matching abs≈sustained first-crossing around t=25–50 (B1 seed2: abs=0, sustained=50). Cleaner “fast recovery” under sustained metrics despite losing to activation on stationary ρ and on `t_star_abs`.

### B3 (pixel permutation) — activation lag

| Seed | \(t^*_{\mathrm{abs}}\) | Recovered? |
|------|------------------------:|------------|
| 0 | 125 | ☑ |
| 1 | 100 | ☑ |
| 2 | 125 | ☑ |

No t=0 first-crossing artifact on B3 activation (abs ≈ sustained at those times). Pixel permutation breaks the frozen-input coincidence that inflates B1 t=0.

| Reporting-only fields | B1 | B3 | Notes |
|-----------------------|----|----|-------|
| Relative recovery \(t^*_{\mathrm{rel}}\) | = abs for activation (all seeds 0) | (see per-seed summaries) | not a pass predicate |
| `gt_shift_disagreement` mean | **0.632** | **0.133** | weak_shift if ≥ 0.90 — neither triggers; B3 is a hard shift |
| Lag window / stride | 500 / 25 | 500 / 25 | from config defaults |

**Lag curve narrative:**

- B1 activation: high ρ at t=0 (artifact) → collapse after real B training starts → genuine climb; sustained recovery ~t=100.
- Residual-energy: shallower post-switch dip; sustained recovery typically ~50.
- Hebbian: late recovery (~300–400).
- GEVB-RF: never recovers under B1 or B3 (0/3 seeds), consistent with stalled FA.
- Artifact paths:  
  `results/historical/pre_patch_rng_isolation/phase0_B1_label_permutation/task_shift/seed_*/time_lag.csv`  
  `results/historical/pre_patch_rng_isolation/phase0_B3_pixel_permutation/task_shift/seed_*/time_lag.csv`  
  `results/phase0_diagnostics/sustained_recovery_report.md`

---

## Fisher diagnostic (post-patch stationary only)

`results/phase0_fisher_stationary/` — no pre-registered Fisher threshold.

Mean late **edge** Spearman (seeds 0–2):

| Candidate | grad_abs | fisher | ablation_relu |
|-----------|---------:|-------:|--------------:|
| activation | 0.732 | **0.675** | 0.199 |
| hebbian_corr | 0.536 | **0.504** | 0.145 |
| eligibility_residual_energy | 0.542 | **0.474** | 0.189 |
| eligibility_gevb_rf | 0.289 | **0.300** | 0.099 |

Activation remains strongest for retention-oriented Fisher as well as current-loss GT.

---

## Digit-split fixed-head run (B2) — scoped

`results/phase0_digit_split/`: A = digits 0–4, B = 5–9→0–4, **fixed 5-way head**.

**What this is:** hard **class-set / distribution replacement** onto the same task shape — not expanding-head class addition (no extra logits / capacity at B).

**What it showed:** activation still wins `late_a` every seed; mean gt_shift ≈ 0.56; sustained recovery cleanest for residual energy; gevb never recovers.

**Still open:** expanding-output-head class addition (orchestrator-relevant growth condition). See consolidated §4.3 / §8.

---

## Interpretation notes

Keep scalar residual-energy (candidates 2–3) separate from GEVB-RF (candidate 4).

### Allowed FAIL / partial interpretations (from design)

| Outcome | Allowed interpretation | Applies? |
|---------|------------------------|----------|
| Candidates 2–3 fail, candidate 4 passes | Scalar residual **energy** is insufficient; vector broadcast is required for Phase 1 info access | ☐ **no** (4 fails; 1 passes) |
| All of 2–4 fail, candidate 1 fails | Local activity + broadcast residual family as designed is not a usable importance proxy | ☐ **no** |
| Candidate 1 ≈ candidates 2–3 | Error term not adding ranking info under scalar path (check candidate 4 before concluding) | ☐ **partially** — residual mid-tier but clearly below activation on edge×grad_abs |
| Stationary pass, recovery fails | `PARTIAL_PASS`: lag is a Phase 1 design constraint (EMA / probe / switch sharpness) | ☐ **no** (recovered under gate) |
| Recovery OK but `weak_shift` | `PASS_WEAK_SHIFT`: B1 label-perm may be vacuous; run B2/B3 contingency before Phase 1 | ☐ **no** (not weak_shift; B3 also PASS with harder shift) |

### Free-form scientific notes

1. **What worked:** `activation` (plain activation magnitude — no broadcast dependence) is a robust, cheap proxy for gradient-based importance: passes stationary threshold on every seed, survives B1 label permutation and B3 pixel permutation, remains best_local under controlled truncated training (post-patch 6 vs 20 ep) and correlates well with Fisher (retention), not just `grad_abs`. Residual-energy eligibility is a competitive mid-tier signal with the cleanest **sustained** lag recovery.
2. **What failed:** `eligibility_gevb_rf` — worst on every measure, every seed, every variant tested. Root-caused to FA rotation never being causally active under backprop training; confirmed via LR × `B_ℓ`-scale sweep (all cells + 40-epoch arm stalled ~90°). Scoped negative: this GEVB-RF *implementation* as a gradient proxy, not “broadcast residual signaling cannot work.”
3. **Candidate-4 caveat:** Do **not** report “broadcast residual proxies cannot work.” `eligibility_residual_energy` (also broadcast-dependent) is legitimate and mid-tier. Only fixed-random feedback GEVB-RF failed as a gradient proxy here.
4. **Sensitivity / follow-ups (do not overwrite primary verdict trees):** truncated-epoch stationary (`results/phase0_truncated_epochs/`), post-patch 20ep baseline (`results/phase0_stationary_20ep_postpatch/`), B3 archive (historical), digit-split fixed-head (`results/phase0_digit_split/`), Fisher (`results/phase0_fisher_stationary/`), GEVB sweep (`results/phase0_gevb_fa_sweep/`), diagnostics (`results/phase0_diagnostics/`).
5. **Pre-patch code / linked experiment ids:** B1 `2026-07-20_0a69524f` and B3 `2026-07-21_c0916ec7` ran **pre-patch** (RNG isolation bug since fixed). Results remain valid for fixed configs; non-bit-reproducible under different edge sample or post-patch code. Full-realism post-patch A/A bit-identical: `results/phase0_profile/full_realism_aa_repro.json`.

### Phase 1 recommendation

| Gate | Decision |
|------|----------|
| `phase1_go` (B1 primary) | **`true`** |
| `phase1_go` (B3 contingency) | **`true`** |
| Recommendation | Proceed to Phase 1 (structural plasticity / orchestrator) design. Use **`activation`** as the primary local importance signal for growth/pruning; monitor **`eligibility_residual_energy`** for shift-recovery dynamics (faster / more genuine under sustained metrics than activation’s `t_star_abs`); **do not** use `eligibility_gevb_rf` without a redesigned training regime (out of Phase 0 scope). Prefer **post-patch** trainers only. Stage Phase 1 incrementally with pre-registered success criteria. Still-open: harder data / reduced capacity, **expanding-head class addition**, harmful-unit detection under `max(0,ΔL)` — documented in consolidated §8; not blockers to *start* Phase 1, but expanding-head addition is better validated once growth mechanics exist. |

---

## Reproduction

```bash
pip install -e ".[dev]"

# Historical primaries are archived — do not overwrite them in place.
# New primary-style full run (new results root / experiment_id):
python scripts/run_phase0.py --config configs/default.yaml --condition both \
  --results-root results/phase0_runs/new_primary

# Stationary-only post-patch baseline:
python scripts/run_phase0.py --config configs/stationary.yaml --condition stationary \
  --seeds 0,1,2 --results-root results/phase0_stationary_20ep_postpatch

# Offline diagnostics on frozen lags / correlations:
python scripts/analyze_sustained_recovery.py
python scripts/analyze_neuron_rankings.py

# CI-scale smoke (not a primary verdict):
python scripts/run_phase0.py --config configs/default.yaml --smoke --condition stationary
```

Confirm historical B1/B3 `meta/experiment_id.txt` and `aggregate/verdict.json` match the tables above before claiming a primary result. For the full post-patch narrative, use
[`docs/phase0_consolidated_results.md`](phase0_consolidated_results.md).
