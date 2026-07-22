# Adaptation-cost — split→restore retention (decisive harder shift)

**Status:** pre-registered before run.  
**Date:** 2026-07-22  
**Runner:** `scripts/run_adaptation_split_restore.py`  
**Results:** `results/phase1a_diagnostics/adaptation_cost/split_restore/`

## Why this cell exists

One-way digit-split and B1 both recover at `t*=1` for **every** arm including
scratch — the post-shift task is easy enough that there is no real adaptation
problem. That cell’s bounded conclusion stands:

> On shifts mild enough that a static net recovers in one epoch, plasticity
> adds no adaptation-cost advantage and ~35% wall-clock overhead.

This document pre-registers **exactly one** harder test — **not** a treadmill —
and treats its outcome as final for the adaptation thesis on this architecture
and these digit tasks.

## Claim under test

**Does plasticity retain task-A capability under interference from task B better
than a fixed network that continues training?**

Not “does plastic relearn A from scratch faster after forgetting,” and not
“does plastic beat one-way shift wall-clock.”

## Protocol

| Phase | Data | Epochs (primary) | Smoke |
|-------|------|-----------------:|------:|
| **A** | Digits 0–4 (identity labels) | 10 | 2 |
| **B** | Digits 5–9 → labels 0–4 | 15 | 3 |
| **C** (restore) | Digits 0–4 again | 15 | 3 |

- Head: 5-way throughout.  
- Seeds: {0, 1, 2} primary; {0} smoke.  
- **No weight reset** at A→B or B→C for primary arms.  
- Plastic: locked stack + self-cal r\* (warm-start on A only) + `max_new=4`.  
- Fixed primary: **frozen_continue** (same SGD lr through A/B/C).  
- Stretch (optional): fixed_finetune (lr×0.1 from B onward).  
- Scratch: **not** a primary retention arm (reinit at B destroys A by construction).

## Metrics (all pre-registered)

### Retention (primary)

After last B epoch, before any C training:

- `acc_A_end` — task-A test acc at end of phase A  
- `acc_A_after_B` — task-A test acc immediately after phase B  
- **`forgetting`** = `acc_A_end - acc_A_after_B`  
- **`retained_acc`** = `acc_A_after_B`

Also log `acc_B_end` (must learn B; both arms should reach ≥ 0.80 on B or the
cell is invalid for retention comparison).

### Restore recovery (secondary)

During phase C, eval on **task A** after each C epoch + switch-into-C eval:

- `acc_rec_bar_A` = `0.90 * acc_A_end`  
- `t_star_sustained` on A (Phase 0 sustained definition)  
- Wall-clock of C until sustained recovery (`T_w_C`) — secondary cost  
- `acc_A_trough_C` — min A acc during C (including C-start)

### Discriminating-power gate (required)

Before claiming any retention result:

- If **both** arms have `retained_acc ≥ 0.85` after B → **TASK_TOO_EASY_FOR_RETENTION**  
  (no catastrophic interference; same failure mode as one-way t*=1).  
- If **both** arms have `retained_acc ≤ 0.25` (chance-level on 5-way) →  
  **BOTH_CATASTROPHIC_FORGET** — retention thesis unsupported; C is pure relearn.  
- Cell is **discriminating** only if arms **differ** on retention by ≥ `δ_ret`
  or one is above 0.5 and the other below 0.35.

`δ_ret = 0.05` absolute accuracy.

## Pass / fail (final for adaptation thesis on these tasks)

Per seed (only if discriminating):

| Label | Condition |
|-------|-----------|
| `RETENTION_PLASTIC_WINS` | `retained_acc_plastic ≥ retained_acc_fixed + δ_ret` |
| `RETENTION_FIXED_WINS` | `retained_acc_fixed ≥ retained_acc_plastic + δ_ret` |
| `RETENTION_TIE` | \|Δ retained\| < δ_ret |

Multi-seed headline (seeds {0,1,2}):

| Headline | Rule |
|----------|------|
| **`RETENTION_PLASTIC_WINS`** | ≥2/3 seeds plastic retention win |
| **`RETENTION_FIXED_WINS`** | ≥2/3 seeds fixed retention win |
| **`RETENTION_TIE`** | ≥2/3 ties, or mixed 1–1–1 |
| **`RETENTION_NON_DISCRIMINATING`** | ≥2/3 seeds hit too-easy or both-catastrophic gates |

**Secondary (report only, does not override retention headline):** restore
`T_w_C` and trough on C — plastic faster restore is interesting only if
retention already favors plastic or is tied.

## Explicit stop rule

This is the **single** harder adaptation cell. After it:

- If plastic retention wins → differentiated claim stands; stop.  
- If tie / fixed wins / non-discriminating → adaptation thesis **not supported**
  on this architecture for these digit tasks; write up honestly; **stop**
  (no further “one more harder shift” without a new product mandate).

One-way digit-split / B1 results remain valid as the **mild-shift** bound.

## Non-goals

- No multi-task joint heads or EWC/replay baselines in this cell  
- No Fashion split-restore in this prereg  
- No retuning rate-cap mid-cell to rescue plastic  
