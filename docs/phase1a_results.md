# Phase 1a Results — Baseline Structural Plasticity

**Status:** **MNIST Pareto `PARETO_BOTH_FAIL` (closed).** Fashion-MNIST dense scout stands; **prior plastic Fashion Pareto is `INVALID_UNTIL_RECALIBRATED_RERUN`** — it reused MNIST-locked `r*_ℓ` without re-probe. Fashion calibration locks **r\* = (0.71, 0.80, 0.91)**; straddle SEPARATES_OK; growth-fire PASS. Artifacts: `pareto_comparison/`, `fashion_mnist_pareto/`, `fashion_mnist_calibration/`.

**Date:** 2026-07-22  

**Pre-registration:** [`phase1a_preregistration.md`](phase1a_preregistration.md)  
**Diagnostic root:** `results/phase1a_diagnostics/`  
**Primary artifacts:** `gap_imprint_survival_{fixed,sliding}/`, `gap_imprint_survival_comparison.json`

---

## Scope (state on every description of this experiment)

Layer structure is fixed (input → H1 → H2 → H3 → output). Only hidden neuron counts and adjacent inter-layer edges are plastic. Locality is sensing radius `R` in a shared unit square — not multi-hop graph diffusion. Long-range false negatives under small `R` are expected.

**Dense-start limitation:** initial topology matches Phase 0 fully connected sizes (`H = [256, 128, 64]`). Early dynamics primarily exercise **pruning** of `init_dense` material. Incentive-based edge growth is implemented and instrumented but **empirically inert** under the freeze trigger/birth (see historical sections).

---

## Headline finding (primary)

### Gap-imprinted growth establishes **and stays specialized** under both fixed and sliding thresholds

Full stack (gated novelty trigger; sep-power edges; open-fan scale; measurement-sized straddle bias). Seeds **{0,1,2}**, 6 ep. Pre-registered longitudinal metrics on **birth gap_mask**.

| Rate (mean seeds 0–2) | Fixed | Sliding |
|----------------------|------:|--------:|
| Birth gate pass | 0.80 | 0.82 |
| Established of births | **0.53** | **0.52** |
| **Stays specialized \| est∧gated** | **0.915** | **0.918** |

Trajectories: gap/whole stays ~**1.8–1.95**, on_off ~**0.34–0.39** through age 5 under **both** arms. Sliding does not help vs fixed. Parent-clone 0% divergence baseline is beaten.

**See §Primary Phase 1a result** for full table, per-seed rates, and supersession of the early τ-sweep A/B comparison.

---

## Historical freeze: under kaiming + act>r*, growth did not establish

*(Retained as the mechanism-chain starting point; superseded as product mechanism by the primary result above.)*

In the fully instrumented 6-epoch diagnostic window (seed 0, `τ=0.5`, grace=3, birth-bias scaffold aligned, **kaiming** birth):

| Quantity | Value |
|----------|------:|
| Neurons grown | 72 |
| Grown neurons pruned | 36 |
| Grown still alive at end | 36 |
| End survivors with age ≥ grace (3) | **0** |
| End survivors that ever passed judgment | **0** |
| Birth→prune lag for all pruned grown | **100% lag = grace** |

**Every** grown unit that reached first contraction-prune eligibility was removed on that step under that design. That non-establishment chain drove the later trigger/birth redesign.

**Artifact:** `results/phase1a_diagnostics/survivor_age_census/summary.json`

---

## Mechanism chain (birth diagnostics, completed)

Six rounds of increasingly precise diagnosis. Each round found something real; each fix was correct for its mechanism. The residual after all of them is still categorical non-establishment — evidence that the floor is not “one more birth knob.”

### Round map

| # | Finding | Action | Outcome |
|---|---------|--------|---------|
| 1 | Target rate mis-set (global / too low) | Lock **per-layer** `r*_ℓ` = (0.77, 1.02, 1.28) from probe | Correct homeostatic baseline |
| 2 | Standalone incentive edge path inert | Keep-strict eligibility locked; seed edges dominate | Scope: 1a does not exercise incentive refill |
| 3 | Thrash = lag-1 newborn cold prune | Grace + EMA warm-start at `r*_ℓ` | Lag shifts 1→2 (then 3 with aligned grace); thrash falls but bar still fails |
| 4 | Raw post during grace ≈ 0; dead-ReLU majority among failures | Temporary **birth-bias scaffold** (0.5, linear decay) | Full-post dead-ReLU → 0%; full-post “alive_attenuated” becomes majority |
| 5 | Grace / bias timers misaligned (judge mid-scaffold) | **grace_steps ≥ bias_steps** (both = 3); enforce in config | Judgment only when scaffold = 0 |
| 6 | Full-post 0.54 was mostly scaffold; weight-only still dark | Decompose weight vs bias; survivor age census | **Cosmetic alive-attenuated; zero establishment** |

### 1. Dead ReLU at birth (real; partially addressed)

**Artifact:** `results/phase1a_diagnostics/newborn_dead_relu/summary.json`  
Without birth bias: ~44% of newborns are dead-ReLU at birth; among later-pruned, ~68%. Alive-attenuated is real but secondary (~33% at birth).

### 2. Grace + EMA (real delay; not a cure)

**Artifact:** `results/phase1a_diagnostics/post_grace/`  
Lag-1 cold death eliminated; thrash reduced (~27% neuron events at the grace=2 checkpoint). **Every** later-pruned grown unit still dies on the **first eligible** step after grace. EMA warm-start only delays judgment while raw activation stays near zero (`newborn_raw_activation/`).

### 3. Birth-bias scaffold (real for full-post class; cosmetic for weight path)

**Artifact:** `results/phase1a_diagnostics/newborn_birth_bias/summary.json`  

| Metric (at birth, full post) | bias=0 | bias=0.5 |
|------------------------------|-------:|---------:|
| dead_relu | 44% | **0%** |
| alive_attenuated | 33% | **98.5%** |
| mean post | 0.052 | **0.54** |

Confirmed: scaffold removes dead-ReLU under **full** post. Superseded by decomposition (next).

### 4. Timer alignment (correctness fix, not leniency)

**Pre-registered:** `newborn_grace_steps = 3` ≥ `newborn_birth_bias_steps = 3`.  
Enforced in `PlasticityConfig.__post_init__`. Judgment eligibility and scaffold zero-crossing coincide.

### 5. Scaffold vs weight-driven decomposition (reveals cosmetic fix)

**Artifact:** `results/phase1a_diagnostics/birth_bias_alignment/summary.json`  

Probe: `post_weight_only = ReLU(Wx + b_param)` vs `post_full = ReLU(Wx + b_param + birth_bias)`. Judgment snapshot is **after** bias decay and **before** prune.

| Window | Cohort | full post | weight-only | birth_bias | dead on weight-only |
|--------|--------|----------:|------------:|-----------:|--------------------:|
| Age 0 | all | 0.534 | **0.065** | 0.50 | 50% |
| Age 0 | later pruned | 0.50 | **0.0** | 0.50 | **100%** |
| Age 2 (last scaffold) | later pruned | 0.167 | **~0** | 0.167 | 94% |
| **Judgment (age≥3, bias=0)** | all who hit eligibility | **~0** | **~0** | **0** | **92%** |

- Age-0 full post ~0.54 is **~88% scaffold** (weight share ~12%).
- Of units classified `alive_attenuated` under full post at birth, **50%** are `dead_relu` on weight-only (**scaffold misclassification**).
- At first unpropped judgment, weight-driven post for the doomed cohort is ~1e−5 — not a developed signal waiting for downstream gain.

**Implication for levers:** hold **outgoing-weight / influence scale** as a thrash fix for the prune majority (nothing real to amplify). Hold further birth-init patches as the default next move; residual after a complete birth-mechanics chain points past local init knobs.

### 6. Categorical non-establishment (headline)

**Artifact:** `results/phase1a_diagnostics/survivor_age_census/summary.json`  
See §Headline. Age-at-end for all 36 “survivors”: only `{0:12, 1:12, 2:12}`. Established grown trajectories past judgment: **n = 0**.

---

## What the τ / thrash numbers actually measured

Pre-registered stability bar (cap hits, 1.5× H bound, thrash ≤ 10) was applied to a grid including `τ ∈ {0.1, 0.2, 0.3, 0.5}`.

**Artifact:** `results/phase1a_diagnostics/tau_sweep_stability/summary.json`

| τ | final H | thrash total | passes thrash bar? | “stable”? |
|--:|--------:|-------------:|:------------------:|:---------:|
| 0.1 | 267 | 697 | no | no |
| 0.2 | 267 | (same band) | no | no |
| 0.3 | … | … | no | no |
| 0.5 | lowest thrash/prune in ranking | … | no | no |

Soft ranking still prefers **τ = 0.5**. **Absolute lock: held.**

**Reframing (mandatory when citing these numbers):** thrash and final hidden totals in this regime are dominated by

1. **Prune dynamics on the dense-init population**, and  
2. A **grace-buffered grow→die conveyor** of unestablished newborns,

**not** by a working grow/prune cycle that successfully refills capacity. Do not interpret τ-grid “stability” as “growth is healthy at this τ.” Do not run further primary-scale τ sweeps or lock τ until the open question below is settled **or** growth-as-cosmetic is an explicit, documented product decision.

Train accuracy in the same short windows is high (informational ≈ **0.98**), consistent with a near-converged dense net under prune — not evidence that plastic growth contributed.

---

## Other locked diagnostics (unchanged scope)

### Target rates `r*_ℓ`

**Artifact:** `results/phase1a_diagnostics/target_rate_probe.json`  
Locked per-layer means: h1 **0.77**, h2 **1.02**, h3 **1.28**. Global scalar rejected.

### Standalone incentive edge path

Live-path `candidates_evaluated = 0` throughout the tested window under keep-strict eligibility + dense start. Observed edge birth is **seed-bootstrap**, not incentive candidacy. Unlocking standalone refill is a **1b+ design question**.

### Edge eligibility

`neither_contraction` (keep-strict) locked for 1a; no new lever after A/B gate probes.

---

## Why non-establishment is a plausible structural stopping point

A newly grown neuron is inserted with **random (Kaiming) weights** into a network that already solves the task at high accuracy. Its incoming directions are uncorrelated with the feature geometry the rest of the net has organized. In that regime, the loss landscape may give the new unit **very little gradient pressure** to become useful: existing pathways already explain most of the variance the objective cares about. Prior logs showed newborn `|grad|` on new rows ~1e−6 during grace (`newborn_raw_activation/`) — **consistent with** this picture but **not yet diagnostic**, because near-converged nets have small grads on established weights too (see step 2 matched-baseline protocol below).

That is a **motivated stopping hypothesis** for *why*, not only *that* — and it is **not** claimed as a fully isolated causal proof in this freeze. It is the reason further birth-mechanics rounds are not the default next spend.

---

## Design gap (biological framing) — trigger/birth decoupling

What is unbiological in the freeze design is **not** “growth without infinite patience.” It is that **biological newborn neurons are never wired with blind, random connectivity disconnected from the reason they were born.**

- New dentate granule cells integrate via the **perforant path** — a structured, information-carrying pathway from entorhinal cortex — not by random attachments hoping usefulness emerges. First inputs already carry task-relevant structure by **where the cell is placed**.
- In the freeze design, the growth trigger (sustained deviation from `r*_ℓ`) can correctly mark a stressed locus, but the newborn then receives **Kaiming-random** weights that carry **zero** information about what help is needed. **Trigger and birth mechanism are decoupled:** one says “this region is stressed,” the other ignores that signal and starts from noise.

Second literature-backed correction: biological newborns do not only get **more time** under adult rules (grace extension already tried and correctly failed). Immature neurons have a **different learning rule** for a critical period (lower LTP threshold, greater plasticity range) before settling into adult rules. That is not “wait longer under the same update” — it is “learn under a temporarily more permissive rule,” with structured placement giving a head start on *what* to learn.

### Concrete mechanism under test (before more gradient-starvation digs)

| Knob | Meaning | Config |
|------|---------|--------|
| **Parent-perturbed birth** | Seed newborn incoming row (+ optional outgoing col) and edge fan as a **noisy copy of the growth-trigger parent**; place coord near parent | `newborn_weight_init: parent_perturbed`, `newborn_parent_noise`, `newborn_copy_outgoing`, `newborn_parent_coord_jitter` |
| **Critical-period LR** | Scale grads on newborn incident weights by `newborn_lr_mult` while age < grace — different rule, not longer grace | `newborn_lr_mult` (1.0 = off) |

Default freeze configs remain `kaiming` / `lr_mult=1.0` until an explicit product flip. Implementation is live (`newborn_weight_init`, `newborn_lr_mult`).

### Parent-seed establishment A/B (6 ep, seed 0, τ=0.5, grace=3)

**Artifact:** `results/phase1a_diagnostics/parent_seed_establishment/summary.json`

| Cell | weight_init | lr_mult | grown | pruned | survivors | **age≥grace (established)** | lag hist |
|------|-------------|--------:|------:|-------:|----------:|----------------------------:|----------|
| kaiming_lr1 | kaiming | 1 | 72 | 36 | 36 | **0** | 100% lag=3 |
| **parent_lr1** | **parent_perturbed** | **1** | 72 | **0** | **72** | **36** | *(none)* |
| parent_lr5 | parent_perturbed | 5 | 72 | 28 | 44 | 8 | lag=3: 28 |
| kaiming_lr5 | kaiming | 5 | 72 | 36 | 36 | **0** | 100% lag=3 |

**Reading:**
- **Parent-perturbed birth alone** breaks categorical non-establishment: **zero grown prunes**, 36 units past grace (ages 3–5 present), full 72 still alive. Couples trigger → structured start; confirms the freeze gap was birth/wiring, not “impossible to establish.”
- **Elevated LR alone (kaiming_lr5)** does nothing — critical-period mult without structured init is not sufficient (matches “different rule needs something to learn,” not magic scale on noise).
- **Parent + lr_mult=5** is **worse** than parent alone (28 lag-3 deaths): permissive rule on an already-hot copy can destabilize. Prefer parent-seed with `lr_mult=1` as the first product candidate; retune critical-period mult only if needed later.
- **Caveat:** establishment under parent copy may partly be **duplicating a high-activation parent** (redundancy) rather than filling a true capacity shortfall. That is still progress on birth/wiring; **trigger-mistarget** remains the right next *diagnostic* (are parents error-loci or just hot?). Do not treat parent-seed as proving the growth *objective* is correct.

**Product implication:** `newborn_weight_init: parent_perturbed` is the leading fix for non-establishment. Config default stays `kaiming` until explicitly flipped so the freeze baseline remains reproducible; primary should not ship kaiming-blind birth as the intended mechanism once this is accepted.

### Establishment ≠ capacity fill — redundancy check (required)

**Artifact:** `results/phase1a_diagnostics/parent_seed_redundancy/summary.json`  
Protocol: parent_perturbed, 6 ep seed 0; track parent→child pairs; weight cosine/corr over age; at end, for age≥grace pairs still alive, **ablate child** and **ablate parent** separately (zero incoming row + bias + outgoing col; probe ΔL = L_abl − L_base, Phase 0–style), plus joint ablate both. Ref: sample of init-origin ΔL for scale.

**n:** 72 pairs born; **36** established (age≥3) with parent still active → full ablation table.

#### 1) Weight similarity (does the clone stay a clone?)

| Age | mean cos(concat in+out) |
|----:|------------------------:|
| 0 (birth) | **0.990** |
| 1 | 0.989 |
| 2 | 0.989 |
| 3 | 0.988 |
| 4 | 0.987 |
| 5 | **0.986** |

- Final cos ≥ 0.9: **100%** of established pairs  
- Cos decay ≥ 0.1 over life: **0%** (max decay in sample ~0.01)  
- Mean cos age0 → final: 0.990 → 0.987  

**Verdict (similarity): REDUNDANCY / non-specialization.** Parent-perturbed newborns **never diverge** into a distinct weight function under this window. “Established” here means “survived judgment as a near-copy,” not “found a new role.”

#### 2) Ablation ΔL (child vs parent)

| Quantity | Value |
|----------|------:|
| mean ΔL⁺ child (relu) | 0.0044 |
| mean ΔL⁺ parent (relu) | 0.0037 |
| mean child/parent (relu) | **0.81** |
| mean ΔL⁺ init-origin ref | 0.0060 |
| frac child near-zero (ΔL⁺ < 1e−4) | 44% |
| frac **both** near-zero | **42%** |
| frac classic “child~0 & parent useful” | **3%** |
| frac child ≥ 0.5× parent (and child > 0) | **53%** |

**Verdict (ablation): not the “dead passenger” pattern.** Children are **not** systematically free (ΔL≈0) while parents carry the loss. Near convergence many units (init and grown) have small ΔL; when ΔL is visible, **child and parent are often on the same scale**. That is still consistent with **duplicate capacity** (two near-identical copies of a useful direction — full parameter cost, limited marginal function), not with **new useful function**. Cosine ~0.99 rules out specialization as the explanation for matched ΔL.

Joint ablation is often **superadditive** vs either alone on hot h2 pairs (e.g. both ≫ child+parent), which is compatible with two aligned copies sharing load rather than orthogonal specialists.

#### Combined reading

| Claim | Status |
|-------|--------|
| Parent-seed → **establishment** (pass grace / not pruned) | **True** (prior A/B) |
| Parent-seed → **genuine new capacity / specialization** | **Not supported** — weights stay clones; ablation does not show independent specialists |
| Risk for Pareto / sparsity | **Live** — growth can inflate H with redundant established copies that pass homeostatic/z-score because they look like the parent |

**Do not flip product default to parent_perturbed on establishment alone.** Treat parent-seed as: solves birth/wiring non-establishment; **does not** pass the “worth the parameters” test.

### Trigger mistarget — **first** (upstream of specialization)

**Artifact:** `results/phase1a_diagnostics/trigger_mistarget/summary.json`  
Why first now: if parents are “hot but already fine,” no inheritance split or decorrelation can create unmet demand that is not there. You would only be tuning how clones form in neighborhoods that never needed a second neuron — exactly the expensive-redundancy-as-stability-win pattern already measured.

**Cohort:** parents of the **36 established** parent-seeded children, metrics at the growth-trigger moment (post-epoch probe). Null: random init-origin active units at the same epochs.

| Metric (established parents, n=36) | Parents | Null active |
|------------------------------------|--------:|------------:|
| mean EMA act | **1.93** | 1.18 |
| mean EMA / r*_ℓ | **1.91×** | 1.12× |
| mean EMA percentile among active | **0.88** | 0.50 |
| mean corr(\|act\|, per-sample CE) | **−0.009** | −0.007 |
| mean CE on top-quartile act samples − CE all | **−0.019** | −0.008 |
| frac corr(act, CE) > 0.05 | **0%** | 7% |
| frac CE_top > CE_all | 28% | 41% |
| frac “genuine demand” (corr>0.05 & CE_top>CE_all) | **0%** | — |
| frac “mistarget” (hot & not error-associated) | **72%** | — |
| train acc at trigger epochs (mean) | **0.94** | — |

**Verdict: `MISTARGET_HOT_BUT_FINE`.**  
Growth triggers on units that are **unusually active** (top ~12% of layer, ~2× r*) in a **near-solved** net, but those units’ activity is **not** aligned with higher local error or uncertainty — if anything, samples where the parent is most on are **easier** (CE_top < CE_all). The homeostatic rule (sustained deviation **above** r*_ℓ) is reading **“running hot / well-engaged”** as **“needs help.”**

**Root-cause order (locked):**

1. **Trigger mistarget** (this result) — fix or re-specify what “need” means (error residual, uncertainty, under-capacity ≠ over-activation).  
2. Only if triggers land on genuine demand: **birth specialization** under structured inheritance (anti-clone).  
3. Parent-seed remains a valid structured-start tool **after** the trigger is fixed — not a substitute for it.

### Growth-trigger candidates — free checks (#2 saturation, #8 novelty)

**Artifact:** `results/phase1a_diagnostics/growth_trigger_candidates/summary.json`

#### #2 Activity saturation (extreme hot tail)

On-disk re-slice of established parents by EMA percentile among active (same mistarget rows):

| Unit band | n | mean corr(act, CE) | mean CE_top−CE_all |
|-----------|--:|-------------------:|-------------------:|
| 75–90 | 15 | −0.005 | −0.013 |
| 90–95 | 12 | −0.014 | −0.031 |
| 95–99 | 5 | −0.026 | −0.029 |
| ≥99 | 1 | +0.001 | −0.001 |
| ≥95 combined | 6 | −0.022 | −0.025 |

Recomputed sample tails (top 1% of **samples** by self |act| for established parents):  
mean CE_top1% − CE_all = **−0.11** (harder? no — **easier**); corr(self, CE) = **−0.02**; 0% with corr > 0.05.

**Verdict: `SATURATION_NULL`.** Narrower extreme-hot bands do **not** recover an error association. #2 is not a growth-trigger fix.

#### #8 Novelty / no strong local neighbor (GNG/ART polarity)

For each parent at trigger: neighbors = other active units within `R=0.35` (same layer). Per probe sample: `nbr_max = max |act|` over neighbors; `novelty = −nbr_max`.

| Metric (established parents, n=36) | Self (hot) | Novelty (low nbr max) |
|------------------------------------|-----------:|----------------------:|
| mean corr with CE | **−0.021** | **+0.091** |
| frac corr > 0.05 | **0%** | **94%** |
| mean CE at top-1% of score − CE all | **−0.11** (easier) | **+0.29** (harder) |
| mean CE when nbr_max bottom-25% − all | — | **+0.10** |
| mean CE gap region (self & nbr low) − all | — | **+0.057** |
| mean CE hot region (self & nbr high) − all | — | **−0.076** |
| frac novelty corr > self corr + 0.05 | — | **92%** |

Null init actives show the same polarity (novelty corr +0.072 vs self −0.012), so this is not an artifact of only “growth-pressure parents.”

**Verdict: `NOVELTY_TRACKS_ERROR`.** Opposite polarity from act > r*: **local coverage gaps** (no strong neighbor response) co-locate with higher CE; hot neighborhoods co-locate with lower CE. Aligns with Growing Neural Gas / ART “spawn where response is weak,” and with existing `missing_in_radius` locality machinery.

#### Pre-implementation pins (density confound + gating)

**Artifact:** `results/phase1a_diagnostics/novelty_density_gate/summary.json`

**1) Density confound — clear (not prune-empty artifact)**

Broad sample stratified by live neighbor count within R (tertiles):

| Density band | mean \|N\| | mean corr(novelty, CE) | CE bot25 nbr_max − all | frac corr>0.05 |
|--------------|----------:|-----------------------:|-----------------------:|---------------:|
| Low | 26 | **+0.059** | **+0.059** | 65% |
| High | **51** | **+0.073** | **+0.077** | **81%** |

Established parents, high-density third (mean **78** live neighbors): corr still **+0.055**, CE gap still **+0.030**. Low-density third also tracks. **Not** “low max only because almost nobody’s left.” Within-unit sample correlations already hold neighbor set fixed, so the signal is activation gap among live peers, not empty set max.

**2) Build gated novelty from the start (not pure novelty)**

Pure novelty can fire in task-irrelevant dead zones (correctly ignored). Offline corr is not a per-site guarantee. Pre-registered rule:

| Component | Definition |
|-----------|------------|
| **Novelty** | High when max \|act\| among active same-layer neighbors in R is low (`−max_j∈N(i) s_j`) |
| **Local CE gate** | Gap is task-relevant: mean CE on bottom-quartile nbr_max samples > mean CE (`local_ce_excess_on_gap > 0`). **Not** self-act > r* |
| **Density floor** | **`k_min = 11`** (measured): require \|N(i)\| ≥ 11 live same-layer neighbors in R |
| **Under-floor** | `|N(i)| < 11` → **excluded** from this mechanism; **logged + monitored** (no silent inert stall). No act>r* or 1b fallback now |

Growth when **novelty ∧ CE gate ∧ density floor**. Rate-cap / free slots unchanged. Birth separate from trigger: **`parent_perturbed`** clones the trigger unit; **`gap_imprinted`** writes CE-excess residual pre-act (not a clone).

Among high-novelty sites in the diagnostic, **100%** already had positive CE excess on the gap — gate is selective against pure dead zones, not vacuous on this window.

**Do not implement ungated pure novelty first** (avoids another minimal-then-patch cycle).

#### Density floor k_min — measured (not a guess)

**Artifact:** `results/phase1a_diagnostics/novelty_kmin_validation/summary.json`  
Stratified sample of active units by live neighbor count \|N\| (n=512 unit-probes across 6 ep). Metrics: local CE excess on gap, novelty–CE corr, half-split sign agreement of CE excess (stability).

| \|N\| bin | n | mean CE excess on gap | std | gate pass % | novelty–CE corr | split sign agree |
|-----------|--:|----------------------:|----:|------------:|----------------:|-----------------:|
| 0 … 5 | 0 | — | — | — | — | — (unobserved; dense-start) |
| **6–10** | 54 | +0.042 | **0.107** | **83%** | **0.047** | 0.83 |
| **11–15** | 81 | **+0.073** | **0.058** | **93%** | **0.071** | **0.88** |
| 16–20 | 93 | +0.092 | 0.051 | 96% | 0.077 | 0.86 |
| 21–30 | 112 | +0.084 | 0.060 | 95% | 0.077 | 0.79 |
| 31–50 | 112 | +0.072 | 0.052 | 95% | 0.068 | 0.83 |
| 51+ | 60 | +0.055 | 0.037 | 93% | 0.053 | 0.73 |

- **Noise cliff:** 6–10 is the weak band (std of CE excess ~2× mid bands; lowest gate-pass and borderline novelty corr).  
- **First fully stable band:** 11–15 and all higher.  
- Cumulative “\|N\| ≥ k” sweep is flat for k∈{0…30} only because almost everyone already has \|N\|≥6; it does **not** validate k_min=0.  
- Top-novelty false-positive rate (CE excess ≤0) is ~0 above any k in this window; the discriminating issue is **estimate stability**, not mean FP.

**Pre-registered:** `k_min = 11` (`|N(i)| ≥ 11`). Optional stricter `k_min = 15` if primary wants higher gate purity. Do not use unmeasured “≥5” or “well below 26.”

#### Under-floor live policy (|N(i)| < k_min) — **pre-registered decision**

Once pruning is live, a unit that previously sat above the floor can fall below it. That is **not** an offline-edge case: h3 medians already trend toward the low teens, so parts of the net can become permanently ineligible for gated-novelty growth if exclusion is absolute.

**Decision (locked before wiring):** regions with `|N(i)| < 11` are **explicitly excluded** from this growth mechanism.

| Choice | Status |
|--------|--------|
| Exclude from gated novelty | **Yes** — hard density floor |
| Log under-floor counts (per layer / step) | **Yes** — required, not optional |
| Monitor fraction of actives under floor (esp. h3) | **Yes** — treat rising under-floor share as a distinct stall signal |
| Fallback to act > r* as floor-of-last-resort | **No** (not now) |
| Fallback via 1b relay / scaffolding | **No** (not now; separate stage if ever needed) |

**Why exclude rather than invent a fallback now:** building act>r* or 1b as an automatic safety net would re-introduce mistarget (hot-but-fine growth) or scope-creep before gated novelty is even measured live. Exclusion is fine **only if it is known and monitored** — not discovered later as “nothing fires, so everything looks fine.” Under-floor stall is a **second kind of stall** (structurally inert, silently), distinct from revolving-door non-establishment and clone-redundancy; logs must make it visible.

**Not claimed:** that exclusion is permanent product policy forever. If live runs show large under-floor fractions killing growth where CE still wants capacity, revisit fallback **explicitly** — do not let one emerge unregistered.

### Gated novelty — implemented + birth-init A/B (6 ep, seed 0)

**Code:** `neuron_growth_trigger=gated_novelty`, `novelty_k_min=11`; probe via `collect_gated_novelty_probe`; under-floor on `StructureEventLog`. Product defaults still `activation` / `kaiming`.

#### Confirmation: what the first battery actually tested

Under gated novelty, `parent_slot=trigger` + `weight_init=parent_perturbed` **copies the trigger unit’s weights/edges** (noisy clone of the unit that *marked* the gap). It is **not** a direction computed from CE-excess samples.

| Mechanism | What is written at birth |
|-----------|---------------------------|
| **`parent_perturbed`** | Clone of trigger unit W/edges + noise; coord near trigger |
| **`gap_imprinted`** | Incoming row = **residual pre-act** on the trigger’s gap samples (CE-weighted mean pre on bottom-quartile nbr_max minus global mean pre), Kaiming-scaled; outgoing Kaiming; **no** parent edge copy (seed-edge path); coord near trigger only |

The first battery tested **old cloning pointed at a new locus**, not content-informed gap init. That was insufficient to conclude “cloning from a neighbor is fundamentally wrong.”

**Artifacts:** `results/phase1a_diagnostics/gated_novelty_battery/summary.json` (now `gap_imprinted`); parent arm via `--weight-init parent_perturbed`.

#### Live numbers (seed 0, 6 ep, gated novelty)

| Check | `parent_perturbed` | **`gap_imprinted`** |
|-------|-------------------:|--------------------:|
| Births / prunes | ~58 / ~136 | 58 / 137 |
| cos(child, **trigger**) @ birth | **~0.99** | **~0.22** |
| cos(child_in, **gap direction**) @ birth | (same as clone) | **~0.98** |
| Dead-ReLU @ birth | 0% dead; ~47% mostly_dead | **0%** dead; **2%** mostly_dead; **98%** alive |
| Established age≥grace | **11** (clones) | **0** |
| cos(trigger) among established | stays ~0.99 | — (none) |
| Under-floor h3 mean frac | ~0.19 | ~0.19 (logged) |

**Readings (ordered):**
1. **Trigger is live** — novelty∧CE∧density fires; under-floor exclusion visible on h3 (~20%), not silent.  
2. **`parent_perturbed` under novelty = clone-at-new-locus** — confirmed. Establishment without specialization is that init’s expected failure, not a test of gap content.  
3. **`gap_imprinted` is genuine content init** — birth cos_gap ~0.98, cos_trigger ~0.22; dead-ReLU cleared cleanly.  
4. **Gap imprint does *not* establish here (n=0 past grace)** — opposite of parent clone’s survival win. We **cannot** claim “content-informed init collapses toward redundancy under ordinary backprop”: units die before multi-epoch convergence to a neighbor can be measured. That stronger dynamics claim is **not supported** and must not be written as if it were.  
5. **Open fork (not pre-locked to anti-clone):** (A) establish gap-imprinted units without re-cloning (critical-period LR, seed/outgoing structure — only with matched baselines); vs (B) keep parent clone for survival and add anti-clone. Do **not** promote “anti-clone birth” alone until (A) is tried or explicitly abandoned.  
6. **Prune own-subset:** whole-probe ΔL understates preferred-class impact for many candidates; absolute levels remain small — no new protective signal from this alone.

**Product:** defaults unchanged. Multi-seed primary + τ lock still **held**.

### Specialist profile + Variant B test (gap-imprinted under fixed vs sliding)

**Hypothesis (user):** a gap-imprinted unit is a **narrow specialist** by construction (high on CE-excess/gap samples, low on the rest). Population-average homeostasis (Variant A, shared r*_ℓ) cannot tell that from “broadly useless,” so death at first judgment may be homogenization working as predicted — not imprint failure. **Check:** before prune, act on own birth-gap samples vs whole probe. If gap ≫ whole while whole ≪ r*, specialist signature confirmed → re-run under Variant B (sliding own-history θ).

**Wiring bugs found while implementing the check (must report — first specialist numbers were invalid):**

1. **Zero edges at birth:** seed-edge path used pre-birth signal snapshot (`s_post=0` for the new slot) → incentive 0 → **no edges**; only `birth_bias` fired → constant post-act. Fixed by warm-starting seed signals / dedicated gap seed.  
2. **Incentive seed ignored imprint content:** even with edges, 4 partners chosen by activity product, not residual magnitude → dense residual mostly masked. Fixed: **`_seed_edges_gap_imprint`** opens top-|W| partners (k_in≥16 on incoming, k_out≥8 outgoing; input has no R filter).

**Artifacts (after wiring fix):**  
- `results/phase1a_diagnostics/gated_novelty_battery_gap_imprinted_fixed/summary.json`  
- `results/phase1a_diagnostics/gated_novelty_battery_gap_imprinted_sliding/summary.json`

#### Specialist profile (gap vs whole) — **does not match the narrow-specialist cartoon**

| Cohort | mean whole | mean gap | mean non-gap | mean gap/whole | frac gap>whole | frac gap/whole≥1.5 | frac specialist signature* |
|--------|-----------:|---------:|-------------:|---------------:|---------------:|-------------------:|---------------------------:|
| **Fixed — at birth** | 0.55 | 0.53 | 0.56 | **0.999** | 47% | **0%** | — |
| **Fixed — last pre-death** | 0.32 | 0.30 | 0.33 | **0.96** | 41% | **0%** | **0%** |
| **Sliding — at birth** | 0.52 | 0.51 | 0.52 | **1.07** | 54% | 6% | — |
| **Sliding — last pre-death** | 0.31 | 0.29 | 0.32 | **0.96** | 30% | 2% | **2%** |

\*specialist signature: `gap/whole ≥ 1.5` **and** `whole < 0.5 × r*_ℓ`.

EMA vs r* at last pre-death: mean **ema − r* ≈ −0.68 (fixed) / −0.74 (sliding)** — low broad average is real; **elevated gap response is not**.

**Verdict: `MIXED_OR_INCONCLUSIVE` → effectively not SPECIALIST_KILLED_BY_AVERAGE.**  
Units die with **low population-average activation** (compatible with contraction under fixed r*), but they are **not** “low overall, high on their target subset.” Mean gap/whole ≈ 1; almost none clear 1.5×. The clean three-part story (trigger finds need → imprint encodes need as specialist → average judgment kills specialist) is **not supported** by the subset activation check. Closer reading: imprint + top-|w| edges produce **real, non-dead** responses, but **not strong sample-selectivity for the birth gap set** under this residual/scale/edge recipe; judgment then correctly sees a unit below r*.

#### Establishment: sliding does **not** rescue gap-imprinted units

| | Fixed (A) | Sliding (B, μ=0.999) |
|--|----------:|---------------------:|
| Births | 64 | 50 |
| Prunes | 135 | **176** |
| Established age≥grace | **7** | **2** |
| Alive at end (any age) | 37 | 18 |

Sliding own-history θ **worsens** thrash/prune load and **does not** lift establishment. With μ=0.999, θ barely moves from r* init while EMA collapses → still looks like contraction for low-average units. Variant B is **not** a free win for gap-imprinted birth in this window.

**Readings:**
1. First “0 established, cos_gap~0.98” battery was **confounded by zero-edge wiring** — constant bias-only activation.  
2. After edge fix, gap-imprint can establish a few units under fixed (7/64) — not categorical zero.  
3. **Homogenization-as-specialist-killer is not the measured mechanism here** — required gap≫whole profile is absent.  
4. **Sliding result is not a completed test of Variant B vs specialists** (see § below) — only “slow θ does not help non-specialists in this window.”  
5. **Why cos~0.98 fails to fire differentially — pre-act diagnostic (next section).** Do **not** lock “turn up imprint selectivity” until that puzzle is resolved.

### Why cos≈0.98 to the gap direction fails to separate gap samples (pre-act diagnostic)

**Puzzle:** weight row is cosine-aligned with residual mean-diff, yet post-act gap/whole ≈ 1. Candidate causes: ReLU dead zone, mean-diff not separating sample clouds, fan-in/edge-mask scale.

**Artifact:** `results/phase1a_diagnostics/gap_imprint_preact_diagnostic/summary.json`  
Protocol: gated novelty + `gap_imprinted`, 3 grow epochs, seed 0, n=36 births. At birth: residual projection of **pre-synaptic** vectors; unit pre/post with and without birth_bias; open-edge residual energy; Cohen's d.

| Quantity (means over births) | Value | Reading |
|------------------------------|------:|---------|
| cos(W_full, residual) | **0.87** | Dense imprint still aligned (noise < 1.0) |
| cos(W_eff, residual) | **−0.005** | **Masked row ≈ orthogonal to residual** |
| Residual energy on open edges | **0.19** | Only ~1/5 of residual support is connected |
| n open incoming edges | 16 | top-\|W\| seed |
| **Ideal Cohen's d: proj(pre_syn, residual) gap vs non-gap** | **+1.03** | **100% of units d>0.3** — mean-diff **does** separate clouds |
| Δ mean proj residual (gap − non-gap) | **+3.12** | Strong linear separation in pre-space |
| Unit Δ mean **pre** (gap − non-gap), no bias | **−0.028** | **Unit pre does not separate** |
| Unit Cohen's d pre | **−0.10** | |
| frac unit pre separates (d>0.3) | **17%** | |
| Unit mean pre (whole), no bias | −0.03 | Near zero / mixed sign |
| frac pre>0 whole | 0.49 | Not globally dead |
| scale: eff std on open / Kaiming-open target | **0.18** | Weights scaled for **full** fan-in, not open fan-in |

**Failure-mode counts (n=36):**

| Mode | n | Meaning |
|------|--:|---------|
| **MASK_OR_SCALE_KILLED_SEPARATION** | **25** | Residual separates samples; unit pre does not |
| **RELU_DEAD_ZONE_OR_SCALE** | 9 | Ideal separates but unit pre collapsed low/negative |
| **RELU_FLATTENS_SEPARATION** | 2 | Pre separates somewhat; post does not |
| SEPARATES_OK / MEAN_DIFF fails | 0 | Mean-diff cloud separation is **not** the failure |

**Resolution of the puzzle:**

1. **Not** “mean-diff direction fails to separate gap from non-gap in pre-space.” Ideal projection Cohen's d ≈ 1.0 for every unit measured. Cosine-to-mean-diff is a valid content signal **when the full vector is applied**.  
2. **Yes, edge mask destroys that separation.** Open edges hold ~19% of residual energy; W_eff is nearly orthogonal to residual in the full ambient space; unit pre Δ reverses or vanishes. Top-\|W\| selection keeps large components but **does not preserve the gap-separating linear functional** once the other ~80% of residual support is zeroed.  
3. **Scale mismatch is real and co-present:** Kaiming RMS targets full fan-in (std~0.067 on h1) while open fan-in of 16 would want std~0.35; measured open std~0.064 (ratio ~0.18). That under-drives pre-act magnitude (feeds ReLU-dead-zone mode for 9/36) but is **secondary** to the mask killing separation (25/36).  
4. **ReLU dead zone alone is not the primary story** for the median unit: frac pre>0 ~0.5 and mean pre near 0, not “both subsets sitting deep below 0 with gap already higher.” Where pre is deeply negative (h2/h3 tails), dead-zone co-occurs **after** mask/scale already failed.

**Implication for “next lever” (pinned before implementing):**  
**Not** “more aggressive imprint selectivity” — mean-diff already separates. Binding break is **realization**, with two distinct sub-causes:

1. **Edge objective was wrong (not only scale).** top-\|W\| picks large residual coefficients, which can be high-dynamic-range dims that are **not** diagnostic of gap vs non-gap. Correct objective: per-dim separating power  
   `score_j = |mean(pre_j|gap) − mean(pre_j|non-gap)| / std(pre_j)`  
   (optionally zeroed when sign opposes residual). Scale-only fix on top-\|W\| edges could still open the wrong support.  
2. **ReLU-dead-zone is a separate bucket** (9 + 2 cases in the first pre-act table). Fixing edge/scale may clear MASK_OR_SCALE without clearing dead-zone; track both through re-measure.

### Realization fix implemented + re-measure

**Code:** `per_dimension_separation_scores` + `_seed_edges_gap_imprint` (select by score_j, not \|W\|) + `_rescale_gap_imprint_open_row` (Kaiming `fan_in = n_open` on residual|_{open}).

**Artifact (post-fix):** same path `gap_imprint_preact_diagnostic/summary.json` (protocol note: `edge_select=per_dim_separation_power`, `scale=open_fan_kaiming`).

| Quantity | Before (top-\|W\| + full-fan scale) | **After (sep-power + open-fan scale)** |
|----------|-----------------------------------:|---------------------------------------:|
| MASK_OR_SCALE_KILLED_SEPARATION | **25 / 36** | **0 / 34** |
| RELU_DEAD_ZONE_OR_SCALE | 9 / 36 | **32 / 34** |
| RELU_FLATTENS_SEPARATION | 2 | 0 |
| SEPARATES_OK | 0 | **2 / 34** |
| Ideal Cohen's d (proj residual) | 1.03 | 1.16 (still 100% d>0.3) |
| **Unit Cohen's d pre** (gap vs non-gap) | **−0.10** | **+1.04** (100% d>0.3) |
| Unit Δ mean pre (gap − non-gap) | −0.03 | **+6.4** |
| scale_ratio (eff std / open Kaiming) | 0.18 | **1.00** |
| mean post gap/whole (no bias) | ~1 | **1.42** (50% ≥ 1.5) |
| mean frac pre>0 | 0.49 | **0.06** |
| mean pre whole (no bias) | ~0 | **−9.8** |

**Readings (post-fix):**
1. **Separation-power edge selection was necessary, not optional.** MASK_OR_SCALE went to **zero**; unit pre now separates as strongly as the ideal residual projection (d≈1.0). That would not be guaranteed by “same top-\|W\| edges, bigger scale” alone.  
2. **Scale fix is in place** (ratio 1.0) and co-required for open-fan RMS.  
3. **Failure mass moved almost entirely into RELU_DEAD_ZONE** (94%). Pre separates *while sitting deep below 0* (mean pre ≈ −10; frac pre>0 ≈ 6%). Gap samples are less negative / more often positive than non-gap, so Cohen's d is large, but **ReLU zeros almost everything** — including much of the gap set on h2/h3. When post is nonzero, gap/whole is often high (mean 1.4, half ≥1.5) — specialist *shape* appears in the tail that clears ReLU.  
4. **Two-bucket tracking confirmed:** edge/scale cleared bucket A; bucket B was operating point.  
5. **Direction formula still held** — ideal and unit pre both separate.

#### Operating-point fix — derived from −9.8, not scaffold 0.5

**Pause before “add a bias”:** mean pre whole **−9.8** with d≈1 is not a small threshold miscalibration. Gap and non-gap sit near **−5 vs −11** (medians similar) — both deep in ReLU’s zero region. A useful constant shift must be **O(5–10)**, not the birth-bias scaffold **0.5** (~10–20× too small). Pure multiplicative weight rescale **cannot** flip a negative mean (s>0 preserves sign).

**Derivation** (`gap_imprint_operating_point_derivation/summary.json`), from the same 34-unit stats:

| Correction | Formula | Mean value over units | Expected after |
|------------|---------|----------------------:|----------------|
| B so mean_gap = 0 | `−mean_pre_gap` | **+5.0** | gap at 0 |
| B so mean_whole = 0 | `−mean_pre_whole` | **+9.8** | whole at 0; gap ≈ +4.8 |
| **B straddle (chosen)** | `−(mean_gap+mean_nongap)/2` | **+8.2** (median 7.1) | gap → **+Δ/2≈+3.2**, nongap → **−Δ/2≈−3.2** |
| Scaffold birth_bias | fixed 0.5 | 0.5 | **insufficient** (ratio ~10–20×) |
| Weight rescale only | s·pre | — | **fails** if mean < 0 |

**Implemented:** per-unit **Parameter** bias  
`B = −(mean_pre_gap + mean_pre_nongap) / 2`  
computed at birth from probe pre on the unit’s open W_eff (same gap mask). Temporary `birth_bias` scaffold **off** for `gap_imprinted` (set 0) so it does not decay against a measurement-sized Parameter bias.

**Re-measure** (`summary_after_straddle_bias.json`, n=36 births):

| Metric | After sep-power+scale (no op bias) | **+ straddle B** |
|--------|-----------------------------------:|-----------------:|
| RELU_DEAD_ZONE | 94% | **0%** |
| SEPARATES_OK | 6% | **100%** |
| mean param B | 0 | **+5.65** (per-unit; matches O(derived)) |
| mean pre whole | −9.8 | **−1.4** |
| mean pre gap / nongap | −5.0 / −11.4 | **+2.79 / −2.77** (straddle as predicted) |
| Cohen's d pre | 1.04 | **0.89** (still strong) |
| frac pre>0 | 0.06 | **0.47** |
| mean post gap/whole | 1.42 | **1.82** |
| frac post gap/whole ≥ 1.5 | 50% | **83%** |
| frac gap samples post>0 | — | **0.72** |
| frac nongap post>0 | — | **0.38** |

**Landed where predicted:** gap cloud above zero, non-gap below; ReLU preserves gap more than non-gap; specialist post shape is now population-scale, not a thin tail.

#### Is straddle the right bias target, or only a convenient one?

Straddle (0.72 gap-on / 0.38 nongap-on) is **not** a binary specialist. Before locking `B=−(μ_gap+μ_nongap)/2`, sweep bias against **post-ReLU** discrimination on the same probe (not pre-mean geometry).

**Artifact:** `results/phase1a_diagnostics/gap_imprint_bias_sweep/summary.json`  
Protocol: n=36 births; for each unit, bias-free pre0; grid `B = B_straddle + linspace(−10, +10, 41)`; score `on_off = frac_gap_on − frac_nongap_on` (and gap/whole).

| Anchor / pick | mean B | on_off | frac gap on | frac nongap on | gap/whole |
|---------------|-------:|-------:|------------:|---------------:|----------:|
| **straddle** | 4.56 | **0.328** | 0.688 | 0.360 | 1.80 |
| gap0 (`−μ_gap`) | 1.10 | 0.300 | 0.534 | 0.234 | **1.96** |
| center whole | 6.28 | 0.323 | 0.748 | 0.425 | 1.73 |
| **best on_off (grid)** | 4.93 | **0.351** | 0.739 | 0.388 | 1.80 |
| best on_off with gap_on≥0.5 | 5.29 | 0.351 | 0.744 | 0.393 | — |

| Check | Value |
|-------|------:|
| mean ΔB (best on_off − straddle) | **+0.38** |
| mean Δon_off (best − straddle) | **+0.023** |
| frac best within **1** of straddle | **61%** |
| frac best within **2** of straddle | **81%** |
| frac straddle within **10%** of best on_off | **81%** |

**Reading:** Maximizing post-ReLU on/off moves B only ~0.4 on average and gains ~0.02 on_off — **not a different operating regime**. gap0 is “cleaner” on gap/whole (1.96) but **worse** on_off and covers only ~53% of gap samples. Straddle is a **near-optimal compromise** for discriminative firing, not an arbitrary geometric convenience. **Lock straddle** as the product formula for now; optional later fine-tune (±1–2) is second-order.

Imperfect 0.72/0.38 is acceptable as a **birth** specialist signature (biology is not binary). The longitudinal question is whether that split **holds after establishment**, not whether birth can be tuned to 1.0/0.0.

#### Pre-registered: Variant B / survival test — “establishes **and stays specialized**”

~~Do not run fixed vs sliding until metrics below are locked.~~ **Run complete** — see **Primary Phase 1a result** below. Spec retained for audit.

**Protocol (pre-registered before any Variant B cell):**

| Item | Spec |
|------|------|
| Stack | gated novelty + `gap_imprinted` (sep-power edges, open-fan scale, **straddle B**) |
| Arms | **fixed** (Variant A) vs **sliding** (Variant B, μ as config default) |
| Window | 6 ep primary diagnostic (extend only if pre-registered) |
| Seed | 0 first; multi-seed only after single-seed readout |
| Birth cohort | all gap_imprinted units with stored birth gap_mask + trigger |

**Primary outcomes (all required):**

1. **Establishment:** alive at age ≥ `newborn_grace_steps` (same bar as prior batteries).  
2. **Birth specialist signature (gate into longitudinal):** at birth, `gap_over_whole_post ≥ 1.3` **and** `on_off = frac_gap_on − frac_nongap_on ≥ 0.2`.  
3. **Stays specialized:** for units that **pass the birth gate and establish**, track on the **same birth gap_mask** each subsequent plasticity epoch until end or prune:

| Trajectory metric | Fail if… |
|-------------------|----------|
| `gap_over_whole_post(t)` | final < **1.2** **or** decay from birth ≥ **0.4** |
| `on_off(t)` | final < **0.15** **or** decay from birth ≥ **0.15** |
| `frac_gap_on(t)`, `frac_nongap_on(t)` | reported for shape; fail soft if nongap_on rises by ≥ **0.2** while gap_on falls by ≥ **0.2** |

Analog of parent-clone cosine table: report **mean trajectories** age 0…T and **continuous** frac of established units that pass stays-specialized (not only a binary headline).

##### Birth-gate clearance on the corrected-straddle cohort (checked, not assumed)

**Artifact:** `results/phase1a_diagnostics/gap_imprint_birth_gate_check/summary.json`  
Gates checked **per unit** (not means only) on the same n=36 straddle batch.

| Source | n | clear **both** | gow≥1.3 | on_off≥0.2 |
|--------|--:|---------------:|--------:|-----------:|
| Bias-sweep straddle scores | 36 | **83%** (30/36) | 97% | 86% |
| Live after orchestrator straddle B | 36 | **89%** (32/36) | 97% | 89% |

Distributions (bias-sweep straddle): on_off median **0.34**, p10 **0.20**, min **0.13**; gap/whole median **1.81**, p10 **1.42**, min **1.29**. Left tail is almost entirely **on_off** (fails sit ~0.13–0.20); gap/whole almost always clears. Failures concentrate in later-epoch **h2/h3** births, not a bulk rejection of imprint success.

**Reading:** Gates sit below population means as intended; **~5/6–8/9 of successfully imprinted units enter the longitudinal count**. Residual left tail is small enough that gate understatement of imprint quality is minor — report `frac_births_passing_gate` alongside trajectories so imprint success is not conflated with fixed-vs-sliding.

##### Stays-specialized “50%” bar — convenience, not measurement-derived

The optional label `SPECIALIST_SURVIVES` uses ≥ **50%** of established units passing stays-specialized as a **round-number convenience** for headline vocabulary only. It is **not** derived the way k_min, sep-power scores, or straddle B were.

- Parent-clone baseline: **0%** weight divergence. Even a **20–30%** stays-specialized pass rate would already be a meaningful, reportable improvement over that baseline.  
- **Primary reportable results** (always publish, regardless of which side of 50%):  
  - `frac_births_passing_birth_gate`  
  - `frac_established` (of births / of gated births)  
  - **`frac_of_established_passing_stays_specialized`** (continuous)  
  - mean `gap/whole` and `on_off` **trajectories** by age  
  - sliding − fixed deltas in **percentage points** on those rates  
- Do **not** bury a 35% pass rate as “fail” solely because it is under 50%, or inflate a 55% as the only success signal.

**Verdict vocabulary (optional binary labels; rates still primary):**

| Label | Meaning |
|-------|---------|
| `ESTABLISH_FAIL` | Almost no age≥grace survivors (either arm) |
| `ESTABLISH_ONLY` | Survivors exist but stays-specialized pass rate is low / trajectories collapse (homogenization after birth) |
| `SPECIALIST_SURVIVES` | Optional: stays-specialized pass rate ≥ **50%** of established — **convenience bar**; always quote the actual rate |
| `B_HELPS_SPECIALIST` | Sliding improves stays-specialized rate (and/or establishment) vs fixed by a clear margin; quote **pp**, not only a 15 pp hard gate |
| `B_NO_HELP` | Sliding does not improve specialist survival metrics vs fixed |

**Out of scope for this cell:** retuning μ, direction formula, or straddle B (unless both arms show ESTABLISH_ONLY with collapsed on_off — then open a new pre-reg).

**Artifacts:** `results/phase1a_diagnostics/gap_imprint_survival_{fixed,sliding}/summary.json`, `gap_imprint_survival_comparison.json`.

---

## Primary Phase 1a result — gap-imprinted specialists under fixed vs sliding

**This is the primary Phase 1a fixed-vs-sliding comparison.** The early τ-sweep A/B (kaiming / act>r* era) is retained only as **infrastructure validation** (cooldown, thrash shape, relative τ ranking). It is **superseded** as a headline mechanism result: trigger, birth, and judgment context have all changed (gated novelty, gap_imprinted stack, specialization metrics). Do not split the document headline across those two eras.

**Protocol (executed):** 6 ep, seeds **{0,1,2}**, stack = gated novelty + `gap_imprinted` (sep-power edges, open-fan scale, straddle B); pre-registered birth gate + stays-specialized trajectory on **birth gap_mask**.

### Mean rates (seeds 0–2)

| Rate | **Fixed (A)** | **Sliding (B)** | Δ sliding−fixed (pp) |
|------|--------------:|----------------:|---------------------:|
| Births pass gate (gow≥1.3 ∧ on_off≥0.2) | **0.798** | **0.817** | +1.9 |
| Established of births (age≥grace) | **0.530** | **0.520** | −1.1 |
| Established of gated | 0.595 | 0.584 | −1.2 |
| **Stays specialized \| established∧gated** | **0.915** | **0.918** | **+0.4** |
| Stays of all births | 0.435 | 0.437 | +0.2 |
| Mean n births / seed | 66 | 69 | — |
| Mean n stays specialized / seed | 28.7 | 30.3 | — |

Per-seed stays|est∧gate: fixed **0.90 / 0.97 / 0.88**; sliding **0.97 / 0.94 / 0.84**. All six cells optional-label `SPECIALIST_SURVIVES` (convenience 50% bar); **primary** is the continuous ~**91–92%** pass rate.

**Why “no help” is a confident read (not a small-delta artifact):** within-arm seed **spread** on stays|est∧gate is **~9 pp** (fixed: 0.88–0.97) and **~13 pp** (sliding: 0.84–0.97). The between-arm mean difference is **~0.4 pp**. Spread-within exceeds difference-between by more than an order of magnitude, so the arms are not resolved as different under ordinary seed noise on this n=3 design — “sliding does not help” is the correct summary of that variance structure, not merely a visually small aggregate delta.

### Longitudinal shape (established ∧ gated; mean over seeds)

| Age | Fixed gap/whole | Fixed on_off | Sliding gap/whole | Sliding on_off |
|----:|----------------:|-------------:|------------------:|---------------:|
| 0 | 1.88 | 0.38 | 1.87 | 0.38 |
| 1 | 1.88 | 0.37 | 1.88 | 0.38 |
| 2 | 1.86 | 0.36 | 1.88 | 0.37 |
| 3 | 1.82 | 0.34 | 1.85 | 0.35 |
| 4 | 1.88 | 0.38 | 1.86 | 0.37 |
| 5 | 1.93 | 0.38 | 1.95 | 0.39 |

**No erosion:** gap/whole stays ~1.8–1.95 and on_off ~0.34–0.39 through age 5 under **both** arms — opposite of parent-clone cosine stasis (clone retained) and opposite of “homogenization after birth” for this signature.

### Reading (pre-registered vocabulary)

| Claim | Status |
|-------|--------|
| Gap-imprint stack produces birth specialists that **establish** | **Yes** (~53% of births; ~59% of gated) |
| Established specialists **stay specialized** | **Yes** (~**91.5–91.8%** of established∧gated) |
| Sliding (B) improves specialist survival vs fixed (A) | **No** — rates match within ~1–2 pp (`B_NO_HELP_SIMILAR`) |
| Parent-clone baseline (0% divergence) | **Beaten** by gap-imprint stays rate ≫ 20–30% |

**Mechanism takeaway:** With a working content birth (trigger + residual realization + measurement-sized operating bias), **shared-r\* fixed threshold does not kill specialists** in this window — the earlier “homogenization kills specialists” failure was dominated by **broken realization** (edges/scale/bias), not by fixed vs sliding judgment. Sliding own-history θ (μ=0.999) is **not required** for specialist survival here and does not add a clear gain.

**Scope honesty:** μ=0.999 is still slow; this does not prove sliding could never help under a faster metaplastic schedule. It **does** complete the pre-registered test: specialists exist, they stay specialized under **both** current arms, and B does not help vs A on the locked metrics.

**Horizon hold (executed):** 6-ep rates were the mechanism primary; **20-ep hold under the locked fixed stack is now measured** (seeds 0–2). See **§20-epoch hold** below — do not silently equate 6-ep and 20-ep numbers.

### 20-epoch hold — establish + stays under locked full stack

**Protocol:** official knobs (`phase1a_fixed.yaml`), fixed threshold only, seeds **{0,1,2}**, **20 ep**, full train steps. Same birth-gate and stays-specialized criteria as the 6-ep primary. Structure snapshots + animation for seed 0.

**Artifact:** `results/phase1a_diagnostics/gap_imprint_survival_20ep_hold/`  
**Visualizations:** `.../visualizations/structure_animation_20ep.gif`, `structure_static_{early,mid,late}.png`

### Mean rates (seeds 0–2)

| Rate | **6-ep primary (fixed)** | **20-ep hold (fixed)** | Δ (pp) |
|------|-------------------------:|-----------------------:|-------:|
| Birth gate | 0.798 | **0.719** | −7.9 |
| **Established of births** | **0.530** | **0.850** | **+32.0** |
| Established of gated | 0.595 | **0.850** | +25.5 |
| **Stays specialized \| est∧gate** | **0.915** | **0.791** | **−12.3** |
| Stays of all births | 0.435 | 0.484 | +4.9 |
| Mean n births / seed | 66 | **234** | — |
| Mean n stays / seed | 28.7 | **113** | — |
| Optional label (all seeds) | SPECIALIST_SURVIVES | **SPECIALIST_SURVIVES** | — |

Per-seed stays|est∧gate (20 ep): **0.750 / 0.805 / 0.818**. Establish of births: **0.860 / 0.846 / 0.845**.

**Right-censoring (do not quote raw alive as establishment):** e.g. seed 0 — alive end 202 = **169** age≥grace (tested) + **33** age&lt;grace (untested late births). Establishment numerator remains **ever age≥grace**.

**How to read the rate shifts (not a failure):**

1. **Establish ↑ (~53% → ~85%):** At 6 ep many births are still inside grace at the cutoff (right-censored). At 20 ep almost all births before the last few epochs get a real judgment window — more units *can* establish, and most do. This is horizon composition, not “6-ep was wrong.”
2. **Stays|est∧gate ↓ (~91% → ~79%):** Longer trajectories give more room for specialization erosion; the continuous rate remains **well above** the 50% convenience bar and every seed is `SPECIALIST_SURVIVES`. Absolute specialist count is higher (mean **113** stays/seed vs ~29 at 6 ep).
3. **Birth gate slightly softer (~80% → ~72%):** More late-life births into a denser, higher-acc net (~0.96–0.97); still the majority pass.

**Verdict:** **`HOLD_OK_RATES_SHIFTED`** — the locked stack **holds at 20 ep** (establish strong; stays specialized dominant). Use **20-ep rates for full-horizon claims**; keep **6-ep rates as the original mechanism primary** (and for fixed-vs-sliding A/B, which was 6 ep only).

#### Cohort re-slice: why stays fell 92%→79% (not assumed benign)

**Question:** is the drop ordinary observation-window softening, or progressive post-establishment erosion (homogenization on a longer clock)?  
**Artifact:** `gap_imprint_survival_20ep_hold/cohort_stays_analysis/` (re-slice of existing trajectories; no new run).

**Cohorts** (among birth-gate passers that ever reach age≥grace):

| Cohort | First age≥grace | n | Final stays|est∧gate | Mean runway after judge |
|--------|-----------------|--:|---------------------:|------------------------:|
| **Early** | by epoch ≤5 (birth ≤2) | 96 | **0.760** | ~15 ep |
| Mid | epochs 6–11 | 140 | 0.729 | ~11 ep |
| **Late** | epochs 12–15 | 85 | **0.859** | ~6 ep |
| Very late | after epoch 15 | 107 | 0.850 | ~1 ep |

**Still-specialized curve (early cohort, vs birth profile):**

| Age | n | frac still passes stays |
|----:|--:|------------------------:|
| 3 (first judgment) | 96 | **0.938** |
| 5 | 78 | **0.949** |
| 10 | 72 | 0.833 |
| 15 | 66 | **0.803** |
| 18 | 47 | **0.723** |

**Matched observation depth (age=5 for every cohort):** early **0.949** vs late **0.937** (Δ ≈ −1 pp) — at the *same* age, early- and late-born specialists look the same.

**Reading:**

1. **Not a birth-cohort quality gap** — early and late match at fixed age. Early units are not “worse births.”
2. **Progressive post-establishment erosion *is* present** — early cohort still-stays falls ~0.95 → ~0.80 → ~0.72 from age 5→15→18. Longer runway reveals drift the 6-ep window largely missed.
3. **Aggregate 79% is runway-weighted** — late/very-late cohorts (~85–86% final stays) had little time to erode; early long-lived units pull the mean down. That is the concerning half of the story, not pure “benign softening that is uniform across cohorts.”
4. **Not a collapse** — even at age 18, early still-stays is **~72%** (≫ 50% bar). Absolute specialist count remains high. But **79% is not a proven equilibrium**: extrapolating the early curve further is unsupported; 40–60 ep could still fall.

**Verdict label:** `HOMOGENIZATION_COMPATIBLE_EARLY_EROSION` — flag explicitly before using a single stays number in the Pareto write-up. Prefer reporting **(a)** 6-ep mechanism stays, **(b)** 20-ep aggregate with cohort curves, **(c)** matched-depth early≈late. Do **not** write “79% stable specialization” without the age-curve caveat.

#### Long-horizon fixed vs sliding retest (the test B was always for)

6-ep `B_NO_HELP` was correctly **held, not failed**: at μ=0.999 sliding never diverged from fixed inside a birth-adjacent window. Progressive early-cohort erosion under fixed reopens the question on a horizon where own-history θ can matter.

**Protocol:** same 20-ep full stack, seeds {0,1,2}, only `threshold_kind` changes (fixed vs sliding, μ=0.999). Same age-binned stays analysis.

**Artifacts:**  
- Sliding hold: `gap_imprint_survival_20ep_hold_sliding/`  
- Comparison: `gap_imprint_20ep_fixed_vs_sliding_cohort/`

| Metric (seeds 0–2) | Fixed 20ep | Sliding 20ep | Δ (S−F) |
|--------------------|----------:|-------------:|--------:|
| Aggregate stays\|est∧gate | 0.791 | **0.829** | **+3.8 pp** |
| Establish of births | 0.850 | 0.842 | −0.8 pp |
| **Early final stays** | **0.760** | **0.820** | **+6.0 pp** |
| Late final stays | 0.859 | 0.804 | −5.5 pp |
| **Early still-stays age 5** | 0.949 | 0.951 | ~0 |
| **Early still-stays age 15** | 0.803 | **0.868** | +6.5 pp |
| **Early still-stays age 18** | 0.723 | **0.829** | +10.6 pp |
| **Early drop age5→15** | **14.6 pp** | **8.3 pp** | **−6.3 pp** |
| Late−early final stays gap | +9.8 pp | **−1.6 pp** | runway skew largely gone |

**Early still-specialized curve (the decisive plot):**

| Age | Fixed | Sliding |
|----:|------:|--------:|
| 5 | 0.949 | 0.951 |
| 10 | 0.833 | 0.893 |
| 15 | 0.803 | **0.868** |
| 18 | 0.723 | **0.829** |

Matched age=5 still early≈late under both arms. Sliding does **not** change birth quality; it **slows post-establishment erosion** of the long-lived early cohort.

**Verdict:** **`B_REDUCES_EARLY_EROSION`** — first positive long-horizon signal for Variant B. Caveats (honest):
1. μ=0.999 is still slow; gains are moderate (+6 pp early final, half the age5→15 drop), not a full freeze of specialization.
2. Aggregate stays only +3.8 pp (late cohorts dominate the mean); the **curve** is the finding, not a large headline delta.
3. This **supersedes “B never helps” as a closed claim** for long-horizon specialist durability. It does **not** alone force a product flip of `phase1a_fixed.yaml` before Pareto — but Pareto write-ups should present fixed vs sliding 20-ep early curves, not only 6-ep B_NO_HELP.
4. Faster μ (if ever tuned) is now a motivated follow-up, not a blind retune.

### Official full-stack recipe (locked)

**Config:** [`configs/phase1a_fixed.yaml`](../configs/phase1a_fixed.yaml) — single product source of truth for primary comparison against the fixed-topology baseline.

| Piece | Locked choice | Basis |
|-------|---------------|--------|
| Threshold | **fixed** | Sliding held (`B_NO_HELP`; within-arm spread ≫ between-arm Δ) |
| Growth trigger | **gated_novelty** (novelty ∧ local CE ∧ \|N\|≥**11**) | Mistarget → novelty chain; under-floor exclude+log |
| Birth | **gap_imprinted** (sep-power edges, open-fan Kaiming, straddle B) | Specialization primary ~53% establish / ~91–92% stays |
| Prune | continuous competitive (z / fraction) | Unchanged; first wave init-dominated |
| Grace / cooldown | grace=**3** ≥ bias_steps=**3**; EMA warm-start at r\*\_ℓ | Cold-prune + judgment alignment |
| r\*\_ℓ / τ | (0.77, 1.02, 1.28) / 0.5 | Locked probe rates; soft τ |

`configs/phase1a_sliding.yaml` is retained as a **held** historical arm with the same birth/trigger knobs for fair re-runs only — not the product recipe.

#### Combined-stack check (pieces together)

Piece-wise batteries never ran the full bundle as one system. Short check (seed 0, 6 ep, full train steps, official YAML):

**Artifact:** `results/phase1a_diagnostics/fullstack_combined_check/`  
**Verdict:** **`FULLSTACK_COMBINED_OK`** (all 10 gates pass)

| Check | Result |
|-------|--------|
| Recipe is locked stack (YAML assert) | pass |
| Gated novelty fires | ~64–70 births / 6 ep |
| Births are gap_imprinted | 100% of grown |
| Gap-seeded edges present | yes (hundreds–thousands) |
| First prune wave init-dominated | epoch 2: **all init** |
| Some gap units establish (age≥grace) | **~50–51%** of births (matches ~53% primary) |
| Accuracy progresses | ~0.95 → ~0.96 |

**Alive-end ≠ establishment (right-censoring — settled).** Same trap as the original survivor-age-census. Example remeasure (`alive_age_split.json`, seed 0, 6 ep):

| Cohort among gap-imprinted | n |
|----------------------------|--:|
| Births | 70 |
| **Ever age≥grace (establishment numerator)** | **36 (51%)** |
| Alive at end | 60 |
| … of which age≥grace (tested, still alive) | **26** |
| … of which age&lt;grace (**right-censored / untested**) | **34** |
| Pruned (all had reached grace) | 10 |

Age histogram of the 60 alive: ages 0/1/2 = 12+10+12 (**34** untested, all born epochs 3–5); ages 3/4/5 = 10+7+9 (**26** tested). So “60 alive” is **not** 60 established — it is 26 tested survivors + 34 too-young-to-judge. The meaningful combined-check establishment figure remains **~50%**, consistent with the primary. Write-ups must not quote raw alive-end as success.

This is **not** a multi-seed primary or 20-ep hold — only “does combining validated pieces still behave sensibly.” It does. Next: multi-seed primary vs fixed-topo baseline under this config.

### Structure visualization (observability tooling)

Opt-in full-state topology dumps + matplotlib renderers. **No effect on training or verdicts**; frozen primaries are not retrofitted.

| Piece | Path |
|-------|------|
| Snapshot export | `experiment.record_structure_snapshots: true` (default **false**); cadence `every_epoch` / `early_mid_late` / epoch list |
| Dump module | `src/neuronca/plasticity/structure_snapshot.py` |
| Static + animation | `scripts/visualize_structure.py` (`--animate` for GIF/MP4) |
| Example run (seed 0, **10 ep**, production growth caps) | `results/phase1a_diagnostics/visualizations/example_seed0/` |

**Example outputs:** `structure_static.png`, `structure_animation.gif` (10 frames), and `structure_static_epoch2_with_prunes.png` (callout). Each subplot is one hidden layer: gray init neurons / low-opacity `init_dense` edges as background; red **gap_imprinted** specialists with dashed **parent-locus** lines to their growth trigger; red `gap_imprinted_seeded` edges at full opacity. Open red circles mark units **pruned this step** (last-living coords captured at deactivation).

**Read the frames correctly (settled):**

1. **`epoch=k` label = post-step state, production growth rate — not a demo inflation.**  
   Snapshot at `epoch=k` is taken **after** epoch-k training **and** that epoch’s orchestrator step. Frame 1 (`epoch=0`) is therefore “after the first plasticity step,” not init-only topology. Growth caps are product defaults (`max_new_neurons_per_layer=4` → up to **12 births/epoch** across h1–h3; `inflated_for_demo: false`). That matches the primary survival battery’s pace (~**66 births / 6 ep ≈ 11/ep** fixed mean). Dense red on frame 1 is expected: ~12 births at the ceiling, each with a full gap-seeded edge fan (always drawn). Only train `max_steps` is shortened for wall-clock; structure caps are not.

2. **Open red “pruned (last frame)” circles — verified on frames 3–10 (epochs 2–9).**  
   Frame-by-frame audit (`frame_verification.json`): markers drawn on **every** epoch with prunes (first fire at **epoch=2** / animation frame 3: **51** open circles, titles `die=6/30/15`). Epochs 0–1 have zero prunes (grace runway) and zero markers — correct negative control. Direct visual check of `frame_0002.png` shows open red rings across all three layers, not only a legend entry. Callout still: `structure_static_epoch2_with_prunes.png`.

3. **Edge draw policy:** **gap_imprinted_seeded and incentive_grown always drawn in full**; the ~500 cap applies to **init_dense only** (`policy` in `*.edge_subsample.json`). Total drawn can exceed 500. Render DPI is 2.5× the first cut (static 350, anim 250).

4. **Epoch-2 prune jump (0→0→51) is 100% init-origin — confirmed.**  
   Origin tags on `pruned_neurons` (`prune_origin_breakdown.json`): epoch 2 cut = **51/51 init**, 0 gap_imprinted; all age=3 (plasticity-step age of the dense-start cohort once histories stabilize). Matches the pre-existing prune-timing story (first competitive cut hits old `init_dense`, not newborns — grace keeps epoch-0 births off the board until later). Cumulative over 10 ep smoke: init 115 (85%), gap_imprinted 17 (13%), kaiming-fallback 3 (2%). Gap deaths appear only from epoch 3 onward and are mostly first-judgment age 3–4; this smoke is **not** a remeasure of the 53%/91% primary (short `max_steps`, no stays-specialized probe), but the first prune wave is cleanly *not* specialists dying early.

Re-run example: `python scripts/run_structure_viz_example.py`. Future primary runs opt in via the config flag and render with:

```bash
python scripts/visualize_structure.py \
  --results results/<run>/stationary/seed_0 \
  --animate -o results/<run>/structure.gif
```

### Relation to early τ-sweep (superseded as headline)

| Era | Trigger / birth | What it measured | Status |
|-----|-----------------|------------------|--------|
| Early τ-sweep A/B | act>r*, kaiming | thrash, relative τ, infrastructure | **Infrastructure validation only** |
| **This run** | gated novelty + gap_imprinted stack | establish **and** stays-specialized | **Primary Phase 1a A/B result** |

### Prune-side activity proxy — ablation ΔL at removal

**Artifact:** `results/phase1a_diagnostics/prune_ablation_mismatch/summary.json`  
Mirror of the growth mistarget question: for units/edges removed by the live competitive prune rule, is ablation ΔL near-zero (activity is a safe “safe to remove” proxy), or is useful structure cut?

Protocol: 6 ep seed 0, kaiming freeze config (dense-init prune dominates). Immediately **before** `deactivate_neuron` / competitive `remove_edge`, measure probe ΔL (neuron: zero in-row+bias+out-col; edge: zero weight+mask). Cap: ≤40 neurons + ≤30 edges ablated per epoch. Control: random **kept** init-origin actives same epochs.

| Cohort | n | mean ΔL⁺ | median | p90 | frac < 1e−4 | frac ≥ 1e−3 | frac ≥ 1e−2 |
|--------|--:|---------:|-------:|----:|------------:|------------:|------------:|
| **Neurons pruned** | 109 | **0.00030** | **0** | 0.0011 | **67%** | **13%** | **0%** |
| Kept neuron control | 72 | **0.0023** | 0.0011 | 0.0059 | 33% | **53%** | 4% |
| **Edges pruned** | 180 | **8e−6** | ~0 | 3e−5 | **99%** | **0%** | 0% |
| **Edges kept (matched control)** | 180 | **1.9e−4** | 6e−6 | 3e−4 | **78%** | **5%** | 0% |

**Neuron-level (trusted):** pruned / kept mean ΔL⁺ ≈ **0.13**. Control’s high impact is distributional (53% of kept clear 1e−3 vs 13% of pruned), not one outlier. Activity is a safe neuron-removal proxy here.

**Edge-level (with control — required):** pruned edges are near-zero **and** surviving edges are also mostly near-zero (78% < 1e−4; p95 still < 1e−3). There is a mild mean gap (pruned/kept ≈ **0.04**) driven by a thin kept tail (max kept ΔL⁺ ≈ 0.0046), but **absolute single-edge ΔL is generically tiny**. That matches Phase 0’s warning: edge-level ablation is a noisy/weak ground truth under dense connectivity. Do **not** read “100% pruned edges near-zero” alone as proof of smart edge selection — the measure has little resolution. Prefer: **neuron prune validated; edge prune not strongly validated by ΔL (measure limited).** Artifact add-on: `control_and_tail_slice.json`.

**13% neuron mismatch tail — origin slice:**

| Origin | n pruned | n mismatch (low-act & ΔL⁺≥1e−3) | rate |
|--------|---------:|--------------------------------:|-----:|
| **init** | 73 | **14** | **19%** |
| **grown** | 36 | **0** | **0%** |

- P(grown \| mismatch) = **0** (base P(grown \| pruned) = 33%) → **enrichment 0**, not grown-concentrated.  
- All mismatch ages ≥ 3 (none inside grace). Ages 3/4/5 only.  
- **Grace extension is not indicated.** Tail is old init-origin low-act units with mild residual ΔL, not unlucky new parent-seeded clones.

**Verdict (updated):**  
- **Neuron competitive prune:** cleared — real pruned-vs-kept separation.  
- **Edge competitive prune:** not cleared as “validated discrimination”; single-edge ΔL lacks resolution (Phase 0-consistent). Mild kept>pruned mean only.  
- **Mismatch tail:** init-dominated, not a grace failure; no new protective-signal design required for grown units.

**Design implication:** growth-trigger redesign remains the primary product lever. Pruning does **not** need a new protective counter-signal on present evidence.

---

## Open follow-ups (priority after mistarget result)

| Priority | Item | Status |
|----------|------|--------|
| **Done** | Trigger mistarget on established parents | **`MISTARGET_HOT_BUT_FINE`** |
| **Done** | Neuron prune ΔL vs kept control | **Validated** (ratio ~0.13; distributional) |
| **Done** | Edge prune ΔL vs kept control | **No strong resolution** — both near-zero (Phase 0-consistent) |
| **Done** | Mismatch tail grown vs init | **Init-only (0 grown)** — grace not the issue |
| **Done** | #2 activity-saturation tail (unit bands + sample top-1%) | **`SATURATION_NULL`** — still no CE link |
| **Done** | #8 novelty / no strong neighbor (max nbr act within R) | **`NOVELTY_TRACKS_ERROR`** — flips mistarget null |
| **Done** | Density confound on novelty | **`DENSITY_CLEAR`** — holds at high \|N\| |
| **Done** | Gating decision | **Gated novelty only** (novelty ∧ CE gate ∧ density floor) |
| **Done** | Density floor k_min | **`k_min = 11`** measured (noise cliff below 11–15) |
| **Done** | Under-floor live policy | **Exclude + log + monitor** (no act>r* / 1b fallback now) |
| **Done** | Implement **gated** novelty growth trigger | Live — `neuron_growth_trigger=gated_novelty` |
| **Done** | Confirm parent_perturbed ≠ gap imprint | **Clone of trigger unit** (cos~0.99); not CE-excess direction |
| **Done** | Implement **`gap_imprinted`** birth | Residual pre-act on gap samples; birth cos_gap~0.98, cos_trigger~0.22 |
| **Done** | Battery: gap_imprinted (pre wiring fix) | Confounded by **zero edges** (bias-only) |
| **Done** | Edge seed fix for gap_imprint (top-\|w\|) | Required for any specialist check |
| **Done** | Specialist profile: gap vs whole act before prune | **No** narrow-specialist signature (gap/whole≈1) |
| **Done** | Pre-act diagnostic (gap vs non-gap proj) | Residual **does** separate (d≈1.0); top-\|W\| mask killed unit pre |
| **Done** | Sep-power edges + open-fan scale | **MASK_OR_SCALE → 0**; unit pre d≈1.0; scale ratio 1.0 |
| **Done** | Dual-bucket tracking | After edges/scale: **RELU_DEAD_ZONE 94%** (mean pre −9.8) |
| **Done** | Derive op-point shift from stats (not scaffold 0.5) | Straddle B ≈ +8 median; scaffold 0.5 is ~10–20× too small |
| **Done** | Per-unit straddle Parameter bias | **RELU_DEAD_ZONE → 0; SEPARATES_OK 100%;** post gap/whole mean **1.82** |
| **Done** | Bias sweep vs post-ReLU on/off | Straddle near-optimal (Δon_off ~0.02; 81% within 10% of best) |
| **Done** | Pre-register survival + **stays specialized** metrics | Locked before run |
| **Done** | **Primary** fixed vs sliding survival (seeds 0–2) | Both ~**91–92%** stays\|est; **B_NO_HELP** (within-arm spread ≫ between-arm Δ) |
| **Done** | Structure visualization (snapshot dump + static/anim) | Opt-in; example under `visualizations/example_seed0/` |
| **Done** | **Lock official full-stack recipe** | `configs/phase1a_fixed.yaml` = fixed + gated_novelty + gap_imprinted |
| **Done** | Combined-stack short check (pieces together) | **`FULLSTACK_COMBINED_OK`** — `fullstack_combined_check/` |
| **Done** | **20-epoch** hold of establish + stays rates | **`HOLD_OK_RATES_SHIFTED`** — est ~85%, stays|est∧gate ~79%; viz in `gap_imprint_survival_20ep_hold/` |
| **Done** | Cohort age-binned stays (fixed 20ep) | **`HOMOGENIZATION_COMPATIBLE_EARLY_EROSION`** |
| **Done** | **20-ep sliding retest** of early-cohort erosion | **`B_REDUCES_EARLY_EROSION`** — early drop age5→15 14.6→8.3 pp; early final stays 0.76→0.82 |
| **Done** | Pre-register Pareto protocol + numeric pass bar | Post-hoc grid; both arms; δ=1.0 pp + n=3 boundary flag |
| **Done** | Run Pareto grid + score both arms | **`PARETO_BOTH_FAIL`** — see §Pareto result; fixed near-boundary |
| **Done** | Dense-grid per-cell seed spread (no retrain) | Median range **0.12 pp**; gaps ≫ dense noise — MNIST fail robust; plateau under plastic P |
| **Done** | **Fashion-MNIST** wide dense scout + seed spreads | Dense flat early (high-80s from ~53k) — scout valid |
| **Done** | **Fashion calibration** (r\* probe + straddle + growth-fire) | r\* **(0.71, 0.80, 0.91)**; **STRADDLE_OK**; growth-fire **PASS**; prior plastic Pareto **INVALID** |
| **Next** | Recalibrated Fashion plastic 20-ep Pareto | Under `configs/phase1a_fashion.yaml` / `plastic_calibrated/` only |
| **Hold** | Faster sliding μ / metaplastic retune | Motivated by erosion only; not MNIST Pareto fix |
| **Hold** | Further MNIST Pareto / δ refinement | **Stopped** — plateau + tight dense noise; no more signal |
| **Hold** | Direction-formula redesign | Pre+post separate; specialists stay specialized |
| **Hold** | #5 correlation complexity | Only if novelty fails as *trigger* (it does not) |
| **Hold** | #4 structural competition | Dense-start confounded until prune has differentiated connectivity |
| **Hold** | #1/#3 local prediction/reconstruction error | Architecture commitment; fallback if cheaper options fail |
| **Hold** | Under-floor fallback | Only if live under-floor frac kills needed growth (h3 already ~20%) |

### Gradient starvation protocol (retained; only if trigger is fixed)

If growth is firing where residual/error says the net actually needs capacity, then non-establishment with near-zero weight-driven post is the structural hypothesis: random-init insert into a near-converged, high-accuracy net gets little loss pressure to become useful.

If growth is firing where residual/error says the net actually needs capacity, then non-establishment with near-zero weight-driven post is the structural hypothesis: random-init insert into a near-converged, high-accuracy net gets little loss pressure to become useful. That is the right place for deeper gradient / learning-path instrumentation — **not** before step 1.

#### Measurement must use a matched baseline (do not skip)

**What is *not* enough:** reporting newborn `|grad|` on the weight row ≈ 10⁻⁶–10⁻⁷ during grace, alone. A network already at >98% accuracy with small CE loss has **small gradients everywhere**, including on established, useful weights, simply because it is near-converged. Comparing a newborn’s gradient to “zero” or to an abstract sense that it “should be bigger” does **not** isolate the hypothesis. That number is consistent with starvation **and** with “late-training grads are small in general” — the latter is true and uninteresting.

**What isolates the hypothesis:** same kind of random init, same “age since random init” axis, **different stage of the host network’s convergence**.

| Cohort | Who | When logged | Host net state |
|--------|-----|-------------|----------------|
| **Established early-life (baseline)** | Sample of **init-origin** rows (or any unit from initial dense birth) | First few epochs after *their* random init — i.e. global **epoch 0–1** of the run (and/or first few hundred train steps) | Rest of net still **learning** |
| **Newborn early-life (treatment)** | Grown units in the current plastic window | First few epochs / steps after *their* birth (age-since-birth 0…grace−1), typically global **epoch 3–5** in the 6-ep diagnostic | Rest of net already **near-converged** |

Both start from the same class of Kaiming-style random init. The only systematic difference is when they arrive relative to host convergence. If established units’ early `|grad|` (at their birth-equivalent age) is **meaningfully larger** than newborns’ early `|grad|` now, that is clean evidence for **“already-converged host starves new units of gradient”** — not merely “gradients are small at the end of training.”

**Units / aggregation (pre-register before the pull):**
- Same statistic for both cohorts: e.g. mean `|∂L/∂W_row|` (incoming row) and optionally mean `|∂L/∂W_col|` (outgoing col), per slot, averaged over a fixed number of train steps at each age-since-init bucket.
- Axis: **age since random init** (plasticity age for grown; step/epoch since run start for init-origin), not wall-clock epoch alone.
- Report: cohort means, medians, and a simple ratio or log-gap `grad_established_early / grad_newborn_early` at matched age buckets (e.g. age 0 and age 1).
- Source for baseline: log in the **same** short plastic run during epoch 0–1 on a fixed sample of init-origin slots (preferred, matched train recipe), **or** reuse Phase 0 / fixed-topo early-training logs only if the train recipe and layer shapes match.

**Interpretive table:**

| Pattern | Reading |
|---------|---------|
| Established early ≫ newborn early (matched age) | Supports **converged-host gradient starvation** |
| Established early ≈ newborn early (both tiny or both comparable) | Does **not** support starvation-as-differential; small grads are generic near optimum / init, or both paths are weak for another reason |
| Newborn early ≫ established early | Unexpected; recheck logging / mask / inactive-grad zeroing |

**Caveat on prior ~1e−6 figure:** `newborn_raw_activation/` reported early mean `|grad|` on new weight rows ~7×10⁻⁷ (later-pruned) / ~3×10⁻⁶ (end-alive). **Do not cite that alone as evidence of starvation** until the matched baseline above is filled.

#### Free logging while running step 1

Step 1 already walks growth triggers during a short plastic train. **In the same code path, log for free** (no extra mechanism work):

1. **Step 1 primary:** parent-neighborhood error / uncertainty / loss contribution at each growth event (+ null active loci).
2. **Step 2 prep (cheap):** per train step (or thin subsample), mean `|grad|` on (a) a fixed sample of **init-origin** rows during **epoch 0–1**, and (b) each **grown** slot during its first grace window, keyed by age-since-init. Write both to the same diagnostic artifact so step 2 does not require a second full train if step 1 comes back clean.

If step 1 says “mistarget,” step 2 prep logs can be archived unused. If step 1 is clean, the matched comparison is already sitting on disk.

### Process note

This ordered pair is an **explicitly named open follow-up**. It is **not** required to close before publishing the non-establishment headline. It **is** required (or a deliberate growth-as-cosmetic decision is required) before absolute `τ` lock and multi-seed primary under the original “working homeostasis” interpretation.

---

## Implications for Phase 1b / 1c (shared risk)

Relay (1b) and scaffolding reuse (1c) will create **new connections and/or new units** that must become useful inside a network that may already be performing well on local subproblems. If weak gradient pressure on random-init attachments into well-performing regions is the structural issue, **those later phases inherit the same risk** — it is not an 1a footnote.

Flag for design docs:

1. Do not assume “grow unit / grow edge → train a few steps → establishment” without an establishment metric.
2. Consider whether growth/rewire triggers should be tied to **task residual / error locality**, not only activation magnitude vs `r*`.
3. Consider whether new structure needs **privileged learning rules**, temporary loss weight, or teacher/scaffold signals stronger than a decaying pre-act bias — or whether growth should be withheld until the net is not already near-ceiling on the objective.

---

## Holds and process locks

| Item | Status |
|------|--------|
| `r*_ℓ` per-layer | **Locked** |
| Edge eligibility keep-strict | **Locked** |
| Grace ≥ bias_steps; birth bias 0.5 / 3 | **Locked** (correctness) |
| Absolute `τ` | **Held** — soft-recommend 0.5; thrash under kaiming measured prune+conveyor |
| Multi-seed primary | **Held** until parent-seed product decision + trigger-mistarget |
| Outgoing-weight scale | **Held** |
| Kaiming as product birth default | Still freeze yaml default |
| Parent-perturbed as product birth default | **Not accepted** — clones + upstream trigger mistarget |
| Growth trigger = act > r* as “need” | **Falsified** for established cohort parents — hot ≠ needed |
| Further blind-random birth patches | **Stopped** |

---

## Primary run template (locked recipe — Pareto pre-registered)

Official plastic arm is the **full stack** in `phase1a_fixed.yaml`. 20-ep hold supplies observed sizes; fixed-topo baselines follow the **pre-registered** grid (not a single guessed match).

| Arm | Config | Results root | Seeds | Status |
|-----|--------|--------------|-------|--------|
| **Plastic full stack (product)** | `configs/phase1a_fixed.yaml` | 20ep hold already in `gap_imprint_survival_20ep_hold/` | 0,1,2 | **Sizes/acc frozen for Pareto** |
| Sliding plastic (required on Pareto) | `phase1a_sliding` knobs | `gap_imprint_survival_20ep_hold_sliding/` | 0,1,2 | **Required arm** — erosion retest on task/param axes |
| Fixed-topology Pareto grid | `phase1a_baseline_fixed_topo.yaml` + H overrides | `results/phase1a_diagnostics/pareto_comparison/fixed_topo_grid/` | 0,1,2 | **Done — both arms scored FAIL** |

### Pareto: matched cost = post-hoc curve (pre-registered)

**Full text:** [`phase1a_preregistration.md` §Pareto comparison](phase1a_preregistration.md) and `results/phase1a_diagnostics/pareto_comparison_prereg.json`.

1. **Not** “pick H and train both arms there” — plastic final P is endogenous (fixed mean ~**88.5k**; sliding mean ~**84.1k**).
2. **Is** run plastic first (fixed **and** sliding, both required) → observe (P, A) → train dense fixed-topo **once** on the pre-declared grid → plot **both** plastic series against that grid.
3. **Why sliding is required:** early-cohort erosion is the failure mode own-history θ targets; 20-ep curves already show B reduces that erosion. Pareto must test whether that shows up as **task-performance-per-parameter**, not only as a specialization diagnostic. 6-ep `B_NO_HELP` never asked this.
4. **Pass bars (locked):** `PARETO_ACC_AT_COST_FIXED` and `PARETO_ACC_AT_COST_SLIDING` — each \(A_p \ge A_b(P_p) - 0.01\) with **δ = 1.0 pp**. Annotate `SLIDING_VS_FIXED_ON_PARETO`.
5. **δ uncertainty (n=3):** if the margin to the bar is **&lt; ~0.5 pp**, flag **`NEAR_BOUNDARY_DELTA_UNCERTAIN`** — seed-spread from three seeds is a noisy estimate of true variance; do not treat a knife-edge as high-confidence. Do **not** retune δ after seeing results.
6. **20-ep accuracy caveat (same as 79% stays):** Pareto reports efficiency **at 20 epochs while specialists are still mid-decline** (cite cohort curves). Not “architecture efficiency” full stop.

### Pareto result (executed)

**Artifact:** `results/phase1a_diagnostics/pareto_comparison/` (`summary.json`, `pareto_curve.png`, `fixed_topo_grid/`)  
**Headline:** **`PARETO_BOTH_FAIL`**

| Arm | A_p | P_p | A_b(P_p) | A_b−δ | margin | Verdict |
|-----|----:|----:|---------:|------:|-------:|---------|
| Fixed plastic | 0.9657 | 88 552 | 0.9780 | 0.9680 | **−0.23 pp** | **FAIL** + **NEAR_BOUNDARY_DELTA_UNCERTAIN** |
| Sliding plastic | 0.9623 | 84 112 | 0.9789 | 0.9689 | **−0.66 pp** | **FAIL** (not near-boundary) |

**Fixed-topo dense grid (mean test acc, seeds 0–2):**

| Tag | params | test acc |
|-----|-------:|---------:|
| tiny [64,32,16] | 53 018 | **0.9725** |
| param_lo [96,48,24] | 81 442 | **0.9795** |
| param_match [112,56,28] | 96 134 | 0.9765 |
| param_hi [128,64,32] | 111 146 | 0.9790 |
| phase0_init [256,128,64] | 242 762 | 0.9805 |
| H_mid / H_hi | 246k / 324k | ~0.979–0.980 |

**Reading (honest):**

1. **Primary predicate fails for both arms** under the pre-registered δ=1.0 pp bar — plastic does **not** match cost-matched dense accuracy within seed-noise allowance.
2. **`DOMINATED_BY_SMALLER_DENSE`:** `param_lo` (81k, 97.95%) beats both plastic means by &gt;1 pp at lower P. Sliding is also dominated by `tiny` (53k, 97.25%, +1.02 pp). Fixed is only +0.68 pp vs tiny (below δ — not annotated as dominate for fixed, but still higher acc at far lower P).
3. **Fixed near-boundary:** margin only −0.23 pp; with n=3 δ uncertainty, do **not** overclaim a “confident deep fail” for fixed — but direction is shortfall, not pass.
4. **Sliding vs fixed on Pareto:** `TRADEOFF_SLIDING_CHEAPER_LOWER_ACC` (−4.4k params, −0.34 pp acc). Reduced early-cohort erosion did **not** yield a dominating task/param point at 20 ep.
5. **Mid-decline caveat still applies:** this is efficiency **at 20 epochs** with specialists still eroding; not a long-run equilibrium claim. A longer-horizon plastic remeasure is a named follow-up if desired — it does not reverse the dense-baseline dominance already visible at smaller P.
6. **δ was not retuned** after seeing results.

#### Dense-grid seed spread (no retrain — weights how hard to read the fail)

**Artifact:** `pareto_comparison/dense_grid_seed_spread.json`

| Tag | params | seed0 | seed1 | seed2 | mean | **range (pp)** | **std (pp)** |
|-----|-------:|------:|------:|------:|-----:|---------------:|-------------:|
| tiny | 53k | 0.9718 | 0.9722 | 0.9734 | 0.9725 | **0.16** | 0.07 |
| param_lo | 81k | 0.9801 | 0.9792 | 0.9791 | 0.9795 | **0.10** | 0.04 |
| param_match | 96k | 0.9770 | 0.9768 | 0.9758 | 0.9765 | **0.12** | 0.05 |
| param_hi | 111k | 0.9806 | 0.9808 | 0.9757 | 0.9790 | **0.51** | 0.24 |
| phase0_init | 243k | 0.9802 | 0.9806 | 0.9806 | 0.9805 | **0.04** | 0.02 |
| H_mid | 246k | 0.9801 | 0.9778 | 0.9798 | 0.9792 | **0.23** | 0.10 |
| H_hi | 324k | 0.9802 | 0.9808 | 0.9803 | 0.9804 | **0.06** | 0.03 |

Median dense within-cell range **0.12 pp** (max **0.51 pp** at param_hi). Plastic fixed seed range was **1.78 pp** — noisier than dense cells.

**Weight on the MNIST result:**

- Gaps dense−plastic (e.g. param_lo **+1.38 pp** vs fixed plastic) are **many×** dense within-cell std (~0.04 pp at that cell). **Not a seed-noise artifact.**
- From ~81k params up, the dense curve is a **flat high plateau** (~97.6–98.1%) with tight seed spreads — plastic P sits under a region with **almost no slope**, so “match the dense frontier” is a hard ask on MNIST regardless of plasticity quality.
- The **cost-matched knife-edge** for fixed (−0.23 pp) remains **NEAR_BOUNDARY** / low confidence as a *precise* fail distance; the **qualitative** story (smaller dense nets already above plastic) is robust.
- **Do not** further refine MNIST δ or re-run MNIST Pareto for confidence.

### Fashion-MNIST

#### Dense scout (still valid — no homeostatic constants involved)

**Artifact:** `results/phase1a_diagnostics/fashion_mnist_pareto/`  
Process: (1) **wide** dense scout not centered on MNIST plastic P; (2) **seed-spread report first**; (3) both plastic arms; (4) cost-matched δ=1.0 pp.

Fashion dense means (20 ep, seeds 0–2) under the same SGD recipe:

| Tag | P | mean acc | range (pp) |
|-----|--:|---------:|-----------:|
| xs_32 | 26k | 0.868 | 1.13 |
| s_64 | 53k | 0.879 | 0.38 |
| m_96 | 81k | 0.878 | 0.67 |
| m_128 | 111k | 0.884 | 0.33 |
| l_256 | 243k | 0.884 | 0.22 |
| xxl_512 | 567k | 0.889 | 0.44 |

- Span **26k→567k**: only **~2.0 pp** total mean-acc gain (0.868→0.889).
- Saturation hint: consecutive gains **&lt;0.3 pp from ~53k** onward.
- Dense seed noise: **median range 0.45 pp**.
- **Finding:** Fashion is **not** a steep dense frontier under this recipe — plateaus early at **high-80s**. Wider grid was the right check.

#### Calibration (required before any plastic Fashion claim)

**Artifact:** `results/phase1a_diagnostics/fashion_mnist_calibration/`  
**Script:** `scripts/run_fashion_mnist_calibration.py`  
**Config:** `configs/phase1a_fashion.yaml`

Every homeostatic constant in the MNIST-locked stack was derived from **MNIST** activation statistics. Porting the same numbers to Fashion is not a transfer test of the mechanism — it is an uncalibrated port. The sequence below is the same one that unlocked Phase 1a on MNIST.

##### 1. Target-rate probe (1 epoch Fashion-MNIST)

| Layer | Fashion r\*_ℓ (mean EMA) | MNIST r\*_ℓ (was) | ratio |
|-------|------------------------:|-----------------:|------:|
| h1 | **0.71** | 0.77 | 0.92× |
| h2 | **0.80** | 1.02 | **0.78×** |
| h3 | **0.91** | 1.28 | **0.71×** |

**Verdict: `INVALID` to reuse MNIST r\*.** h2/h3 are **22–29% lower**. Under MNIST targets on Fashion activations, **~66–69%** of h2/h3 units sit *below* r\* at probe end → chronic contraction pressure and newborn warm-start at the wrong set-point. That is the same class of bug as the early r\*=0.1 miscalibration (near-universal thrash), not a small offset.

##### 2. Straddle operating point (re-measured, formula shape transfers)

Per-unit formula **B = −(μ_gap+μ_nongap)/2** is computed at birth from the live probe (not a frozen MNIST scalar). Re-measure under Fashion r\* + full stack (n=62 births, 6 ep, seed 0):

| | Fashion | MNIST ref |
|--|--------:|----------:|
| bias-free mean_pre_gap | **+5.29** | ≈ −5.0 |
| bias-free mean_pre_nongap | **−13.17** | ≈ −11.4 |
| mean Δ (gap−nongap) | **+18.5** | ≈ +6.4 |
| after B: mean pre gap / nongap | **+9.23 / −9.23** | +2.8 / −2.8 |
| SEPARATES_OK | **100%** | 100% |
| RELU_DEAD_ZONE | **0%** | 0% |
| birth-gate both (gow≥1.3 ∧ on_off≥0.2) | **90%** | ~89% |

**Reading:** Fashion’s bias-free clouds are **not** the MNIST −5/−11 regime — gap means already sit positive after sep-power+open-fan, with a **much larger** gap/nongap separation. The **formula shape transfers**; the **numeric operating point does not**. Scaffold 0.5 remains ~wrong-order-of-magnitude vs realized pre. **Verdict: `STRADDLE_OK`.**

##### 3. Growth-fire gate (under Fashion r\*)

6 ep, seed 0, fixed + gated_novelty + gap_imprinted, Fashion r\*:

| Metric | Value |
|--------|------:|
| neurons grown / pruned | 62 / 24 |
| steps with growth | 6/6 |
| final test acc | **0.804** |
| max consecutive acc swing | 11.2 pp |
| growth_fire_gate_pass | **true** |
| unstable_swings_flag | true (still; reduced severity vs uncalibrated 20-ep) |

Growth fires every epoch and births clear the imprint gate. Mid-training swings are **not fully gone** at 6 ep.

##### Pre-20ep checks (executed — do not skip)

**Artifact:** `results/phase1a_diagnostics/fashion_mnist_calibration/PRE_20EP_CHECKS.md`

**1. Swing concentration (6ep vs 12ep mid-hold).** At 6ep the 11.2pp hit looks one-shot (ep1→ep2, 54% of swing mass, co-timed with first mass prune n=13). Extending to **12ep falsifies that reading**: further crashes at ep5→ep6 (**−25.4pp**, prune only 3) and ep8→ep9 (**−27.2pp**, prune **0**) are not prune-timed. 7/11 transitions exceed 5pp; late-4 max = 27pp. **Verdict: `RECURRING_INSTABILITY_AFTER_RSTAR_FIX`.** r\* was necessary, not sufficient.

**k_min=11 is not the second miscalibration.** Mean \|N\| stays **52–59** (≫11); max frac \|N\|<11 only **3.7%**; under-floor excludes ~2–4% of H. **`KMIN_NOT_THE_RESIDUAL`** — no k_min re-probe before 20ep.

**2. Recalibrated plastic P vs dense scout.** Active P declines steadily **231k→134k** over 12ep (edge prune; fill 0.93→0.44) while H grows 460→561. Late slope projects **P@20ep ≈ 79k** — same ballpark as uncal ~91k, not off-grid. Nearest dense cells **m_160 (142k)** / **m_128 (111k)** sit on the flat high-80s plateau. **`SCOUT_GRID_STILL_RELEVANT`** — no grid extension; score near 80–140k anchors.

**Go/no-go (updated after fixed-topo control):** r\* OK · straddle OK · k_min OK · dense grid OK · **thrash is orchestrator-specific, not Fashion+SGD.**

##### Fixed-topo control (executed — names the residual)

**Artifact:** `results/phase1a_diagnostics/fashion_mnist_calibration/FIXED_VS_PLASTIC_VOLATILITY.md`

Same Fashion data + SGD, 12ep, dense MLP 256-128-64 (seeds 0–2) and frozen-mask DynamicMLP (seed 0). **No orchestrator.**

| Condition | max consecutive val \|Δ\| | sum \|Δ\| | #Δ>5pp |
|-----------|-------------------------:|----------:|-------:|
| Fixed MLP (mean over seeds) | **~1.3 pp** | ~5.6 pp | **0** |
| Frozen DynamicMLP s0 | ~1.1 pp | ~6.5 pp | 0 |
| Plastic Fashion-r\* s0 | **27.2 pp** | **126 pp** | **7** |

Plastic / fixed (s0) ≈ **23–25×** on max and sum swing. Static nets climb smoothly into the high-80s (dense-scout regime). Plastic oscillates 0.46–0.84 under the **same optimizer and data**.

**Verdict: `ORCHESTRATOR_SPECIFIC_THRASH`.** Hypothesis B (task+SGD volatility) is **ruled out as primary**. Residual instability requires **live structure change**.

**Leading (not yet isolated) drivers** — constants never Fashion-reprobed:

1. **Edge prune** `prune_edge_fraction=0.05` (fill 0.93→0.44 / 12ep; fits late crashes with neuron-prune≈0)
2. **Growth insertion** up to 12 gap-imprinted births/ep (late crashes often grown=12, pruned=0)
3. **Neuron z** `prune_neuron_z=-1.0` (early ep2 co-timed with mass prune; **cannot alone** explain prune=0 crashes; within-layer z is scale-free so wider gap Δ does not auto-break it)

**Before 20ep efficiency:** cheap grow/prune 2×2 ablate (full / grow-only / prune-only / frozen) on Fashion r\*, seed 0, 8–12ep. Do **not** treat a thrashing 20ep final acc as a clean efficiency ceiling without that isolation or an explicit decision to accept orchestrator thrash as product behavior.

#### Prior uncalibrated plastic Pareto — retracted as a mechanism finding

| Arm | mean A_p | mean P_p | Verdict (historical) |
|-----|---------:|---------:|----------------------|
| Fixed (MNIST r\*) | 0.661 | 90.2k | **INVALID** (miscalibrated r\*) |
| Sliding (MNIST r\*) | 0.646 | 90.2k | **INVALID** (miscalibrated r\*) |

Those runs used `target_rates = (0.77, 1.02, 1.28)` on Fashion data. Unstable mid-training swings (~0.5–0.85) are the signature of **thresholds fighting the data**, not a clean “mechanism generalizes to a lower ceiling.” A smoothly-converged-but-lower-accuracy result would be a real generalization finding; the thrash is not.

**Headline for plastic Fashion efficiency:** **`INVALID_UNTIL_RECALIBRATED_RERUN`**. Dense scout findings (flat frontier) remain. Recalibrated plastic lives under `fashion_mnist_pareto/plastic_calibrated/` once re-run (`configs/phase1a_fashion.yaml`, `scripts/run_fashion_mnist_pareto.py`).

**Mechanism vs product framing:** MNIST specialization and MNIST Pareto stand. Fashion product claims need Fashion-calibrated homeostasis first.

---

## Artifact index (diagnostic freeze)

| Artifact | Path |
|----------|------|
| Target-rate probe (MNIST) | `results/phase1a_diagnostics/target_rate_probe.json` |
| Target-rate probe (Fashion) | `results/phase1a_diagnostics/fashion_mnist_calibration/target_rate_probe.json` |
| Fashion calibration (full) | `results/phase1a_diagnostics/fashion_mnist_calibration/` |
| Structure liveness | `results/phase1a_diagnostics/structure_liveness.json` |
| Growth-fire / pre-cap | `results/phase1a_diagnostics/growth_fire_gate/` |
| Eligibility A/B | `results/phase1a_diagnostics/eligibility_gate_ab/` |
| τ sweep (raw) | `results/phase1a_diagnostics/tau_sweep/` |
| τ stability bar | `results/phase1a_diagnostics/tau_sweep_stability/` |
| Thrash detail | `results/phase1a_diagnostics/tau_thrash_detail/` |
| Post-grace thrash | `results/phase1a_diagnostics/post_grace/` |
| Newborn cold prune | `results/phase1a_diagnostics/newborn_cold_prune/` |
| Raw activation in grace | `results/phase1a_diagnostics/newborn_raw_activation/` |
| Dead-ReLU split | `results/phase1a_diagnostics/newborn_dead_relu/` |
| Birth bias A/B | `results/phase1a_diagnostics/newborn_birth_bias/` |
| Weight vs bias + judgment | `results/phase1a_diagnostics/birth_bias_alignment/` |
| **Survivor age census (kaiming headline)** | `results/phase1a_diagnostics/survivor_age_census/` |
| **Parent-seed establishment A/B** | `results/phase1a_diagnostics/parent_seed_establishment/` |
| **Parent-seed redundancy (ablation + sim)** | `results/phase1a_diagnostics/parent_seed_redundancy/` |
| **Trigger mistarget (established parents)** | `results/phase1a_diagnostics/trigger_mistarget/` |
| **Prune-time ablation ΔL (activity proxy)** | `results/phase1a_diagnostics/prune_ablation_mismatch/` |
| **Trigger candidates #2 saturation / #8 novelty** | `results/phase1a_diagnostics/growth_trigger_candidates/` |
| **Novelty density confound + gated prereg** | `results/phase1a_diagnostics/novelty_density_gate/` |
| **k_min density-floor validation** | `results/phase1a_diagnostics/novelty_kmin_validation/` |
| **Gated novelty battery (live)** | `results/phase1a_diagnostics/gated_novelty_battery/` |

---

## One-paragraph abstract (for citations / Phase 1 handoff)

Phase 1a implements neuron-level homeostasis on a dense-start MLP. Under **Kaiming** birth, zero grown units permanently establish. **Parent-perturbed** birth establishes but yields **clones** (parent–child weight cos ~0.99; duplicate capacity, not new function). On the parents of those established clones, the growth trigger is **mistargeted**: parents are hot (~2× r*, ~88th EMA percentile) in a ~94% acc net, yet activity is **not** tied to higher local CE/error (mean corr≈0; CE when parent is most active is **below** mean CE). The rule “grow where act > r*” confuses well-engaged units with under-capacity. Expensive redundant clones of already-fine neighborhoods are the expected outcome. **Next lever: redefine need (error/uncertainty residual), not further birth-init tuning.** Structured birth remains useful only after triggers mark real demand. `τ` lock and multi-seed primary held.
