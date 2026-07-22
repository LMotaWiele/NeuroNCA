# Adaptation-cost / task-shift — pre-registration

**Status:** pre-registered before primary runs.  
**Date:** 2026-07-22  
**Depends on:** locked Phase 1a stack (gated novelty + gap_imprinted + competitive prune; self-cal `r*_ℓ` warm-start; Fashion cadence cap=1 when on Fashion).  
**Runner:** `scripts/run_adaptation_cost.py`  
**Results root:** `results/phase1a_diagnostics/adaptation_cost/`

This document freezes baselines, cost axes, recovery metrics, shifts, and
pass/fail interpretation **before** headline runs. Do not retune bars after
seeing primary multi-seed outcomes.

---

## 1. Claim under test

Does **structural plasticity** reduce the **cost of adapting** when the task
changes, relative to a fixed-topology network that keeps training under the same
optimizer recipe — not “does plastic beat training from scratch,” and not
“does plastic match stationary Pareto.”

---

## 2. Baselines (what fixed-topo is allowed to do at the shift)

| Baseline | At switch | Role |
|----------|-----------|------|
| **`frozen_continue`** | Keep weights; continue SGD on post-shift data (same lr/momentum/wd) | **Primary** — honest “task changed under a normal net” |
| **`finetune_low_lr`** | Keep weights; continue with **lower LR** (factor `finetune_lr_mult`, default 0.1) | **Stretch** — stronger CL-style control; report if primary is close |
| **`retrain_scratch`** | Reinit weights; train only on post-shift data for `epochs_b` | **Deprioritized** — low bar; do not lead write-ups with this arm |

**Plastic arm:** same switch as primary baseline (no weight reset). Structure may
grow/prune through A and B under the locked orchestrator.

**Not allowed for primary claim:** comparing plastic-continue only to
retrain-from-scratch, or silently resetting the fixed net’s optimizer state in
a way plastic does not also get.

Optimizer state: **continue** for `frozen_continue` and plastic (no scrub at
switch). `retrain_scratch` builds a fresh optimizer after reinit.

---

## 3. Shifts

| Shift | Primary? | Head | Notes |
|-------|:--------:|------|--------|
| **Digit-split (B2)** | **Yes** | 5-way | A: digits 0–4; B: digits 5–9 → 0–4. Changes what must be computed — no frozen-weight free lunch at switch |
| **Label permutation (B1)** | Secondary / cross-check | 10-way | **Frozen-weight artifact:** activations unchanged at π-switch; switch-instant and first-crossing recovery **not trusted**. Report with explicit flag |

Dataset for primary: **MNIST** (digit-split is defined on digit classes). Fashion
adaptation-cost is out of scope for this prereg (optional later).

---

## 4. Cost axes

| Axis | Role | Definition |
|------|------|------------|
| **Wall-clock (s)** | **Primary cost** | Process time from start of phase B training through sustained recovery (or through full B if never recovered) |
| **Train steps** | Secondary | Optimizer steps into phase B until sustained recovery (or full B) |
| **Epochs** | Secondary only | Epochs into B until sustained recovery — **not** the product cost headline |

**Rationale:** Plastic steps pay extra structure work (probe + orch.step + mask
edits). Epochs hide that overhead. Product claim is cost of adaptation →
**wall-clock primary**, steps reported alongside, epochs footnote.

Also log (not pass gates): cumulative `neurons_grown` / `neurons_pruned` /
`edges_pruned_competitive` in B; mean wall-clock per epoch B.

---

## 5. Metrics (accuracy / recovery)

All recovery metrics use **task-B validation accuracy** (test or held-out probe
on B distribution), logged:

1. **Immediately after switch**, before any B optimizer step (`t=0` / switch eval)
2. **After each B epoch** (`t = 1 … epochs_b` in epoch units; also record
   cumulative B train steps and wall-clock)

### 5.1 Pre-switch reference

- `acc_pre` = val acc on **task A** at end of phase A (last A epoch).
- For digit-split, A and B use different inputs/labels — `acc_pre` is **not** a
  B-domain score. Recovery targets are defined on **B-domain** trajectories only
  (see §5.2). Report `acc_pre` for context.

### 5.2 Recovery level (absolute on B)

Because digit-split B is a new class set, there is no meaningful “return to
acc_pre on B.” Pre-register:

- **`acc_rec_bar` = 0.80** absolute test acc on task B (5-way MNIST digit-split),
  chosen as “clearly above chance (0.2) and in the usable zone under this SGD
  recipe.”  
- Secondary report: fraction of best B acc achieved by fixed_continue at end of
  B (`acc / acc_end_fixed`), for relative reading only — not the primary gate.

For **B1** (secondary): `acc_rec_bar = 0.90 * acc_pre` (same domain after π).

### 5.3 `t_star_sustained` (primary recovery time)

Reuse Phase 0 definition, applied to **accuracy series after switch**:

- Let `t` index post-switch eval points in **epoch units after switch**
  (`t=0` switch eval, `t=1` after first B epoch, …).
- `t_star_abs` = first `t` with `acc(t) ≥ acc_rec_bar` (diagnostic only).
- **`t_star_sustained`** = first `t` such that **all** logged `acc(u)` for
  `u ∈ [t, t_end]` satisfy `acc(u) ≥ acc_rec_bar`, or `+∞` if none.
- **`recovered_sustained`** iff `t_star_sustained` finite.

**Primary recovery time for cost:** wall-clock from start of B training until
the eval that achieves `t_star_sustained` (if `t_star_sustained=0`, cost = 0 B
train — rare on digit-split; on B1 do **not** treat t=0 as success without
sustained hold through B).

Do **not** use first-crossing alone for pass/fail.

### 5.4 Trough depth (peak degradation)

- **`acc_trough`** = min post-switch B val acc over `{t=0 … t_end}`.
- **`trough_depth`** = `acc_rec_bar - acc_trough` (positive = how far below
  recovery bar the worst point sits).  
  Also report `1 - acc_trough` and `acc_pre - acc_trough` where domain-matched
  (B1 only for the latter).

Links to Fashion stability: a fast recovery that craters mid-B is not an
automatic win.

### 5.5 End-of-B accuracy

- `acc_end` = B val acc after final B epoch.
- Secondary to sustained recovery + trough + cost.

### 5.6 Structure (plastic only)

- Hidden counts and active param count at end A / end B.
- Cumulative grow/prune in B.
- Optional: specialist stays under shift deferred (not required for first cell).

---

## 6. Protocol

### 6.1 Stack (locked)

| Knob | Value |
|------|--------|
| Growth trigger | gated_novelty |
| Birth | gap_imprinted |
| Prune | competitive (frozen `prune_neuron_z=-1.0`) |
| r\*_ℓ | **self_calibrate.r_star=true**, warm-start 1-ep probe on phase-A data |
| Rate cap | `max_new_neurons_per_layer=4` on MNIST digit-split primary (Fashion cap=1 is Fashion-specific; MNIST product remains 4 unless retuned) |
| Optimizer | SGD lr=0.01, mom=0.9, wd=1e-4, batch=128 |
| Init size | 256-128-64 (plastic max 512-256-128) |
| Fixed topo size | **match plastic init** 256-128-64 (not post-hoc final plastic size) for primary cell |

### 6.2 Horizon

| Phase | Epochs (primary) | Smoke |
|-------|-----------------:|------:|
| A (pre-shift) | 10 | 2 |
| B (post-shift) | 15 | 3 |
| Seeds | {0, 1, 2} primary | {0} smoke |

### 6.3 Arms (primary cell)

1. `plastic` + digit-split + frozen_continue switch  
2. `fixed_continue` + digit-split  
3. (optional same seeds) `fixed_finetune`  
4. (optional, deprioritized) `fixed_scratch`

---

## 7. Pass / fail interpretation (pre-registered)

### 7.1 Per-seed classification

For each seed, after full A+B:

| Label | Condition |
|-------|-----------|
| `BOTH_FAIL_RECOVER` | neither arm `recovered_sustained` |
| `FIXED_ONLY` | fixed recovered, plastic did not |
| `PLASTIC_ONLY` | plastic recovered, fixed did not |
| `BOTH_RECOVER` | both recovered_sustained |

When `BOTH_RECOVER`, compare **wall-clock to sustained recovery** (`T_w`):

| Sub-label | Condition |
|-----------|-----------|
| `PLASTIC_FASTER_COMPUTE` | `T_w_plastic < T_w_fixed * (1 - δ_T)` |
| `FIXED_FASTER_COMPUTE` | `T_w_fixed < T_w_plastic * (1 - δ_T)` |
| `COMPUTE_TIE` | otherwise |

**δ_T = 0.10** (10% wall-clock margin).

Trough guard (applies to any plastic “win” claim):

- Plastic must **not** have `acc_trough` worse than fixed by more than **δ_trough = 0.10** absolute acc  
  (i.e. reject win if `acc_trough_fixed - acc_trough_plastic > 0.10`).

### 7.2 Multi-seed headline (seeds {0,1,2})

| Headline | Rule |
|----------|------|
| **`ADAPT_COST_PLASTIC_WINS`** | ≥2/3 seeds `PLASTIC_ONLY` or (`BOTH_RECOVER` ∧ `PLASTIC_FASTER_COMPUTE` ∧ trough guard) |
| **`ADAPT_COST_FIXED_WINS`** | ≥2/3 seeds `FIXED_ONLY` or (`BOTH_RECOVER` ∧ `FIXED_FASTER_COMPUTE`) |
| **`ADAPT_COST_TRADEOFF`** | mixed seeds, or plastic faster compute but fails trough guard on ≥2/3 |
| **`ADAPT_COST_BOTH_FAIL`** | ≥2/3 `BOTH_FAIL_RECOVER` |
| **`ADAPT_COST_INCONCLUSIVE`** | anything else / seed noise on wall-clock |

Always publish the full table: `T_w`, steps, epochs, `t_star_sustained`,
`acc_trough`, `acc_end`, structure churn — never a single number without trough.

### 7.3 Explicit non-claims

- Beating `retrain_scratch` alone ≠ adaptation-cost win.  
- Epochs-faster with worse wall-clock ≠ win.  
- B1 first-crossing or switch-instant recovery ≠ win.  
- Stationary Pareto success/failure is out of scope here.

---

## 8. B1 secondary cell (if run)

Same baselines and cost axes. Differences:

- `acc_rec_bar = 0.90 * acc_pre`  
- Flag every report: **B1 frozen-weight artifact** — do not interpret
  `t_star_abs=0` or high switch-instant acc as recovery.  
- Headline uses `t_star_sustained` only.  
- Does not override digit-split primary headline.

---

## 9. Smoke vs primary

| | Smoke | Primary |
|--|------:|--------:|
| Seeds | 0 | 0,1,2 |
| epochs_a / epochs_b | 2 / 3 | 10 / 15 |
| Arms | plastic + fixed_continue | + optional finetune |
| Shift | digit_split | digit_split |

Smoke validates wiring and metric JSON shape only — no scientific claim.

---

## 10. Artifact layout

```text
results/phase1a_diagnostics/adaptation_cost/
  prereg.json                 # machine-readable bars (this doc)
  digit_split/
    seed_{k}/
      plastic.json
      fixed_continue.json
      [fixed_finetune.json]
      [fixed_scratch.json]
    summary.json
    README.md
  b1_label_perm/              # secondary if run
    ...
```

---

## 11. Implementation notes (runner)

- Reuse `first_sustained_crossing` from `neuronca.eval.time_lag` on
  `(times, accs)` with `threshold=acc_rec_bar`.  
- Digit-split loaders: filter indices 0–4 / 5–9; head `num_classes=5`; flip
  `TaskContext` to `DigitSplitTransform(5,9)` at switch for B loaders.  
- Self-cal: one-epoch probe on **phase A** train before plastic A; warm-start
  r\*.  
- Wall-clock: `time.perf_counter()` around B train only for `T_w` (also log
  full-run clock).  
- Fixed and plastic use identical batch size, epoch counts, and eval cadence.
