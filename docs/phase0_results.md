# Phase 0 Results Write-Up

Post-run scientific summary for **Local Importance Signal Validation**.
Fill placeholders after a primary multi-seed run (`scripts/run_phase0.py` / `scripts/reproduce.sh`).
Do **not** redefine signal formulas or thresholds after seeing results; archive under a new experiment id if measurement code changes.

| Field | Value |
|-------|-------|
| **Experiment id** | `<!-- meta/experiment_id.txt -->` |
| **Config hash / ledger** | `<!-- first8(sha256(config.resolved.yaml)) -->` |
| **Git commit** | `<!-- git rev-parse HEAD -->` |
| **Date (UTC)** | `<!-- YYYY-MM-DD -->` |
| **Seeds** | `0, 1, 2` (or note deviation) |
| **Conditions** | stationary · task_shift |
| **Results root** | `results/phase0/` (or `results/phase0_runs/<experiment_id>/`) |
| **Primary artifacts** | `aggregate/verdict.json`, `meta/thresholds.json`, `meta/config.resolved.yaml` |

---

## Verdict summary

| Field | Value |
|-------|-------|
| **verdict** | `PASS` / `PASS_WEAK_SHIFT` / `PARTIAL_PASS` / `FAIL` |
| **phase1_go** | `true` only if `verdict == PASS` |
| **best_local** | `<!-- e.g. eligibility_gevb_rf -->` |
| **stationary_pass** | `true` / `false` |
| **recovered** | `true` / `false` |
| **weak_shift** | `true` / `false` |
| **catastrophic_fail** | `true` / `false` |

Paste key fields from `aggregate/verdict.json`:

```json
{
  "verdict": null,
  "phase1_go": null,
  "best_local": null,
  "stationary": {},
  "task_shift": {}
}
```

---

## Pass / partial / fail criteria

Pre-declared thresholds (copy from `meta/thresholds.json`; do not edit post-hoc).

| Criterion | Metric | Threshold (declared) | Observed | Met? |
|-----------|--------|----------------------|----------|------|
| **Stationary pass (primary)** | Best local **edge** Spearman vs `grad_abs` at stationary `late`: mean ρ ≥ threshold **and** ≥ 2/3 seeds ≥ seed floor | mean ≥ **0.60**; seed floor **0.50** (min count **2**) | mean=`___`; seeds above floor=`___`/3 | ☐ |
| **Stationary secondary** | Same best local, **neuron** Spearman vs `ablation_relu` at `late` | mean ρ ≥ **0.50** (informational only) | `___` | ☐ n/a |
| **Shift recovery** | Same `best_local`, edge vs `grad_abs`: finite \(t^*_{\mathrm{abs}}\) ≤ **500** for enough seeds | ≥ **2 of 3** seeds recover | seeds recovered=`___`/3; \(t^*\) means=`___` | ☐ |
| **Partial pass** | Stationary pass; shift recovery fails | lag is a Phase 1 design constraint | — | ☐ |
| **Fail** | Stationary pass not met | rethink info access; see FAIL interpretation | — | ☐ |
| **Catastrophic fail (annotation)** | \(\max_c \bar\rho(c) <\) `fail_rho_ceiling` at stationary late | ceiling **0.30** | max mean ρ=`___` | ☐ |
| **Weak shift** | Mean `gt_shift_disagreement` (ρ_shift) ≥ ceiling | ceiling **0.90** → `PASS_WEAK_SHIFT` if recovered | ρ_shift mean=`___` | ☐ |

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

| Candidate | ρ seed 0 | ρ seed 1 | ρ seed 2 | mean ρ | Notes |
|-----------|----------|----------|----------|--------|-------|
| `activation` | | | | | |
| `hebbian_corr` | | | | | |
| `eligibility_residual_energy` | | | | | |
| `eligibility_gevb_rf` | | | | | |

`best_local` = argmax mean ρ (ties: prefer GEVB-RF → residual-energy eligibility → hebbian_corr → activation).

---

## Per-layer notes

Primary pass/fail uses **connection-level** (edge) Spearman. Per-layer breakdowns are mandatory so aggregate ρ does not hide layer failures.

| Layer | best_local edge ρ (stationary late, mean over seeds) | Notes (collapse / noise / rank invert) |
|-------|------------------------------------------------------|----------------------------------------|
| `fc1` | | |
| `fc2` | | |
| `fc3` / `fc_out` | | |
| overall (z-score concat) | | |

Secondary / diagnostic:

| Pair | Layer scope | Observed ρ | Comment |
|------|-------------|------------|---------|
| best_local vs `grad_abs` (edge) | per-layer | | primary |
| best_local vs `ablation_relu` (neuron) | per-layer | | secondary |
| other candidates vs GT | overall / layer | | optional |

**Observations:**

- <!-- Which layers drive or break overall ρ? -->
- <!-- Any layer where scalar residual-energy candidates diverge from GEVB-RF? -->
- <!-- Snapshot isolation / probe task correctness checks: green? -->

---

## Time-lag notes

Task-shift recovery uses the **global** `best_local` (frozen from stationary) vs `grad_abs` on the dense lag window.

| Seed | \(t^*_{\mathrm{abs}}\) (first ρ ≥ 0.60 within 500) | \(\rho^{\mathrm{rec}}\) (first-crossing or ρ@500) | \(\rho_{\mathrm{pre-switch}}\) (`late_a`) | \(\rho_{\mathrm{max}}\) in window | Recovered? |
|------|-----------------------------------------------------|---------------------------------------------------|-------------------------------------------|-----------------------------------|------------|
| 0 | | | | | ☐ |
| 1 | | | | | ☐ |
| 2 | | | | | ☐ |
| **Aggregate** | steps mean / median = | `recovery_rho_mean` = | | | `recovered` = |

| Reporting-only fields | Value | Notes |
|-----------------------|-------|-------|
| Relative recovery \(t^*_{\mathrm{rel}}\) | | not a pass predicate |
| `gt_shift_disagreement` mean | | weak_shift if ≥ 0.90 |
| Lag window / stride used | 500 / 25 (default) | confirm from config |

**Lag curve narrative:**

- <!-- Immediate post-switch drop: how sharp? -->
- <!-- Monotonic recovery vs non-monotonic (first-crossing matters)? -->
- <!-- Seed disagreement on recovery time? -->
- <!-- Artifact paths: `task_shift/seed_*/time_lag.csv` -->

---

## Interpretation notes

Fill after primary run. Keep scalar residual-energy (candidates 2–3) separate from GEVB-RF (candidate 4).

### Allowed FAIL / partial interpretations (from design)

| Outcome | Allowed interpretation | Applies? |
|---------|------------------------|----------|
| Candidates 2–3 fail, candidate 4 passes | Scalar residual **energy** is insufficient; vector broadcast is required for Phase 1 info access | ☐ |
| All of 2–4 fail, candidate 1 fails | Local activity + broadcast residual family as designed is not a usable importance proxy | ☐ |
| Candidate 1 ≈ candidates 2–3 | Error term not adding ranking info under scalar path (check candidate 4 before concluding) | ☐ |
| Stationary pass, recovery fails | `PARTIAL_PASS`: lag is a Phase 1 design constraint (EMA / probe / switch sharpness) | ☐ |
| Recovery OK but `weak_shift` | `PASS_WEAK_SHIFT`: B1 label-perm may be vacuous; run B2/B3 contingency before Phase 1 | ☐ |

### Free-form scientific notes

1. **What worked:** <!-- -->
2. **What failed:** <!-- -->
3. **Candidate-4 caveat:** <!-- Do not report “broadcast residual proxies cannot work” from scalar-only FAIL -->
4. **Sensitivity / follow-ups:** <!-- under `results/phase0/sensitivity/` only; never overwrite primary verdict -->
5. **Invalidated runs / linked experiment ids:** <!-- bugfix after primary → new id -->

### Phase 1 recommendation

| Gate | Decision |
|------|----------|
| `phase1_go` | `true` / `false` |
| Recommendation | Proceed to orchestrator design **only** if `phase1_go`; otherwise redesign information access or re-run contingency shift (B2/B3) under a new experiment id |

---

## Reproduction

```bash
pip install -e ".[dev]"
# Primary full run (ledger freezes thresholds + config hash):
bash scripts/reproduce.sh
# Or explicit:
python scripts/run_phase0.py --config configs/default.yaml --condition both
# CI-scale smoke (not a primary verdict):
python scripts/run_phase0.py --config configs/default.yaml --smoke --condition stationary
```

Confirm commit, `meta/experiment_id.txt`, and `aggregate/verdict.json` match the tables above before claiming a primary result.
