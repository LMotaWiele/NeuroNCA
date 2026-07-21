# Phase 1a Pre-Registration

**Status:** **pre-primary diagnostic freeze** — `r*_ℓ` locked, keep-strict locked; birth-mechanics chain complete. **Headline:** growth does not permanently establish (see [`phase1a_results.md`](phase1a_results.md)). **`τ` not absolute-locked**; multi-seed primary **held** until the named follow-up — **(1) trigger mistarget first, (2) gradient starvation only if (1) clean** — or an explicit growth-as-cosmetic decision. Do not treat thrash/τ grid numbers as evidence of a working grow/prune cycle.

**Scope statement (permanent for 1a write-ups):** Layer structure is fixed (input → H1 → H2 → H3 → output). Neuron counts and adjacent inter-layer edges are plastic. Locality uses sensing radius `R` in a shared unit square. Long-range false negatives under small `R` are an expected, measured result. **Dense-start limitation:** initial topology is Phase 0 fully connected, so early training primarily exercises pruning. **Growth limitation (diagnostic freeze):** under current settings, grown neurons form a grace-buffered revolving door and do not permanently establish; final hidden size is not evidence of successful refill.

**Neuron-level homeostasis** (growth/contraction pressure from activation vs `r*_ℓ`, competitive prune, rate-limited neuron growth) is live and is the primary structural mechanism tested in 1a.

**Standalone incentive-driven edge formation** is implemented and instrumented. In the tested window (once-per-epoch grow, dense start, keep-strict eligibility), it is **empirically inert**: every epoch has live-path `growth_no_contraction = 0` and `candidates_evaluated = 0` — not “rare,” a clean **zero**. Observed edge birth is dominated by the **birth-seed bootstrap** path. Do not treat seed-grown edges as evidence that distance-cost / incentive wiring was exercised. Unlocking standalone refill is a **named design question for 1b+** (relay will touch the same candidacy machinery).

## Locked formulas

| Item | Value |
|------|--------|
| Distance | Euclidean in unit square |
| Decay | linear: `decay(d) = 1 - d/R` for `d ∈ [0, R]`, else not a candidate |
| Incentive | `s_i * s_j * decay(d)` |
| Edge eligibility | Pressure checks **only on hidden endpoints**; fixed input/output always eligible. ≥1 hidden growth pressure; 0 hidden contraction pressure |
| Variant A edge bar | `incentive > τ` |
| Variant B edge bar | eligibility + `incentive > max(θ_g)` over growth-pressure hidden endpoints (not `θ_i·θ_j`) |
| Sliding θ | `θ ← μ·θ + (1-μ)·s` from own history only |
| Cadence | once per epoch (after training steps) |
| Step order | update θ → prune edges → prune neurons → grow edges → grow neurons |
| Cooldown | freed slot/edge ineligible until next plasticity step |
| Optimizer on (re)activate | reinit weights + zero optimizer state for touched indices |

## Hyperparameters

| Parameter | Value | Notes |
|-----------|--------|--------|
| `R` | `0.35` | |
| Initial H | `[256, 128, 64]` dense `init_dense` | Phase 0 match |
| Cap `H_max` | `[512, 256, 128]` | 2× safety bound |
| Target rates `r*_ℓ` | **per-layer, probe means:** h1=`0.77`, h2=`1.02`, h3=`1.28` | See §Target-rate probe. **No global scalar.** |
| Variant A `τ` | **soft-recommend `0.5`; not absolute-locked** | See §τ sweep (stability bar); thrash fails all grid cells |
| Edge eligibility | **`neither_contraction` (keep-strict)** | Standalone incentive path inert in tested window; no new lever for 1a |
| Newborn grace | **`newborn_grace_steps = 3`** | Min age before contraction prune; **≥ `newborn_birth_bias_steps`** so judgment never sees residual scaffold bias |
| Newborn EMA init | **`layer_target`** | Seed activation neuron EMA at that layer’s `r*_ℓ` on grow (not 0) |
| Newborn birth bias | **`newborn_birth_bias = 0.5`**, decay over **`newborn_birth_bias_steps = 3`** ages | Temporary pre-act scaffold (buffer, not Parameter); linear decay to 0; fully gone when grace ends |
| Newborn weight init | **Default freeze: `kaiming`**. Leading fix: **`parent_perturbed`** | Parent = growth-trigger slot; copy incident weights + edge fan + near-parent coord (`newborn_parent_noise=0.1`). See establishment A/B |
| Newborn critical-period LR | **`newborn_lr_mult = 1.0`** (off) in freeze; optional >1 | Scales grads on newborn rows/cols while age < grace; **not** a substitute for parent-seed (kaiming+lr5 still zero establishment) |

### Birth bias result (dead-ReLU lever)

**Artifact:** `results/phase1a_diagnostics/newborn_birth_bias/summary.json`  
6 ep, seed 0, τ=0.5, grace=2, EMA warm-start; bias 0 vs 0.5.

| Metric | bias=0 | **bias=0.5** |
|--------|-------:|-------------:|
| dead_relu at birth (all) | 44% | **0%** |
| dead_relu at birth (later pruned) | 68% | **0%** |
| mostly_dead at birth (all) | 23% | **1.5%** |
| alive_attenuated at birth (all) | 33% | **98.5%** |
| mean post at birth | 0.052 | **0.54** |
| Grown pruned count | 37/61 | 38/66 |

**Confirmed (full-post only):** birth bias eliminates the dead-ReLU-at-birth majority under the **full** (weight+scaffold) post.  
**Superseded by weight-vs-bias decomposition below:** that “alive_attenuated majority” was largely scaffold-propped; do **not** treat full-post 0.54 as evidence that weight-driven signal is already healthy.

### Grace / birth-bias timer alignment + weight-vs-bias decomposition

**Pre-registered choice (correctness, not leniency):** raise `newborn_grace_steps` to **3** so `grace_steps ≥ birth_bias_steps`. Judgment is eligible only when age ≥ 3, at which `update_birth_biases` has already set scaffold bias to **0**. Enforced in `PlasticityConfig.__post_init__`.

**Artifact:** `results/phase1a_diagnostics/birth_bias_alignment/summary.json`  
6 ep, seed 0, τ=0.5, grace=3, bias_steps=3, bias=0.5. Decomposition: `post_weight_only = ReLU(Wx+b_param)` vs `post_full = ReLU(Wx+b_param+birth_bias)`. Judgment probe is taken **after** bias decay and **before** prune.

| Window | Cohort | mean post full | mean post weight-only | mean birth_bias | frac dead weight-only |
|--------|--------|---------------:|----------------------:|----------------:|----------------------:|
| Age 0 (birth) | all | 0.534 | **0.065** | 0.50 | 50% |
| Age 0 | later pruned | 0.50 | **0.0** | 0.50 | **100%** |
| Age 2 (last scaffold) | later pruned | 0.167 | **~0** | 0.167 | 94% |
| Age 2 | not-yet-pruned survivors | 0.370 | **0.214** | 0.167 | 25% |
| **Judgment (age≥3, bias=0)** | **all who hit eligibility** | **~0** | **~0** | **0.0** | **92%** |

Additional facts:
- Birth→prune lag hist: **100% lag=3** (first eligible step).
- Of 72 units classified `alive_attenuated` under **full** post at birth, **50%** are `dead_relu` on weight-only (scaffold misclassification).
- Every unit that reached first judgment was later pruned; end-of-run “survivors” had not yet aged to grace (late births) or still carry real weight-driven signal at age 2 (~0.21).

**Verdict: WEIGHT_DRIVEN_STILL_NEAR_ZERO for the pruned cohort.**  
Age-0 full post ~0.54 was **~88% scaffold** (weight share ~12%). At the first unpropped judgment, mean weight-only post is ~1e−5 — not a developed signal waiting for downstream gain.

### Survivor age census — growth never permanently establishes (6-ep window)

**Artifact:** `results/phase1a_diagnostics/survivor_age_census/summary.json`  
Same protocol (6 ep, seed 0, τ=0.5, grace=3, bias_steps=3). Question: of end-of-window grown “survivors,” how many are past judgment age?

| Quantity | Value |
|----------|------:|
| Grown total | 72 |
| Pruned (all lag=3) | 36 |
| Still alive at end | 36 |
| Survivors with **age_at_end < 3** (grace) | **36 / 36** |
| Survivors with **age_at_end ≥ 3** | **0** |
| Survivors that ever `passed_judgment` | **0** |
| Age-at-end hist | age0: 12, age1: 12, age2: 12 |

Every end survivor was born in epochs 3–5 (still inside the grace buffer). Every unit that *did* reach first eligibility was pruned on that step. **Established grown trajectories: n=0** — nothing to decompose past judgment among survivors.

**Final composition (ep 5):** active 378 = **342 init-origin** + **36 grown-origin** (grown share ~9.5%). Epoch trace of grown-alive is a flat **revolving door at 36** (12/layer × grace window): +12 grow / −(init + lag-3 grown) each step; init-origin falls 448 → 342.

**Headline for 1a framing:** under current settings in this window, **zero grown neurons permanently establish themselves**. Hidden-layer sizes tracked in the τ sweep are sustained by (1) incomplete reduction of the original dense-init population and (2) a grace-protected queue of unjudged newborns — **not** by successful new growth. The growth half of homeostatic plasticity is currently **non-functional** (not merely noisy). Read sparsity, Pareto, and stability metrics with that scope constraint.

**Write-up:** full chain in [`phase1a_results.md`](phase1a_results.md). **Primary Phase 1a result (6 ep):** gap_imprinted stack, fixed vs sliding, seeds 0–2 — establish ~53%, stays-specialized ~91–92% both arms; **B_NO_HELP** (within-arm seed spread ~9–13 pp ≫ between-arm ~0.4 pp). Supersedes early τ-sweep as headline A/B.

**Official product recipe (locked for now):** [`configs/phase1a_fixed.yaml`](../configs/phase1a_fixed.yaml) = fixed + gated_novelty + gap_imprinted. Combined check: **`FULLSTACK_COMBINED_OK`**. **20-ep holds** done; sliding reduces early erosion. **Pareto (executed):** **`PARETO_BOTH_FAIL`** — plastic fixed/sliding short of cost-matched dense acc under δ=1.0 pp; dominated by smaller dense cells (e.g. tiny 53k @ 97.3%). Fixed near-boundary (−0.23 pp). Sliding cheaper but lower acc (trade-off, not dominance). Artifact: `pareto_comparison/`. 20-ep = mid-decline snapshot.

### Newborn grace + EMA warm-start — measured outcome

**Pre-registered mechanism (implemented; grace later raised to 3 for bias alignment):**  
(1) `newborn_grace_steps` — contraction prune requires `plasticity_step - birth_step ≥ grace` (now **3**).  
(2) `newborn_ema_init=layer_target` — on grow, set `ActivationMagnitude` neuron EMA at slot to that layer’s `r*_ℓ`.

**Artifact:** `results/phase1a_diagnostics/post_grace/`  
Before/after at τ=0.5, 6 ep seed 0:

| | neuron thrash | edge thrash | total | birth→prune lag hist | thrash ≤10? |
|--|-------------:|------------:|------:|----------------------|:-----------:|
| Old (grace=0, EMA=0) | 120 | 567 | 687 | **100% lag=1** | no |
| New (grace=2, EMA=r*) | **88** | **472** | **560** | **100% lag=2** | no |

**What fixed:** lag-1 newborn death is gone; thrash reduced (~27% neuron events).  
**What remains:** every grown slot that is later pruned is still pruned on the **first eligible step after grace** (lag=2 exactly). Warm-start at `r*_ℓ` + 2 epochs of training is not enough to keep newborns out of (contraction ∧ z<−1) when grace ends — the cycle is delayed, not broken. Absolute thrash bar still fails for the whole τ grid.

### Raw activation during grace (why EMA warm-start fails)

**Artifact:** `results/phase1a_diagnostics/newborn_raw_activation/summary.json`  
Protocol: 6 ep, seed 0, τ=0.5, grace=2, `newborn_ema_init=layer_target`. Log **raw** mean `|act|` (batch) and EMA for each grown slot during age 0–1; compare to established init-origin units.

| Quantity | Value |
|----------|------:|
| Grown with traces | 52 (40 later pruned, 12 still alive at end) |
| EMA at birth (seeded) | **~1.02** (matches r*) |
| Raw mean\|act\| at **age 0** | **0.048** (median **0.0**) |
| EMA at age 0 | ~0.71 (decaying from seed) |
| Raw mean\|act\| at **age 1** | **0.0036** (median **0.0**) |
| EMA at age 1 | **~0.026** (collapsed) |
| Established init raw (ref) | h1 **0.80**, h2 **1.21**, h3 **1.76** |
| Incoming weight row mean\|w\| at birth | ~0.058 (Kaiming scale) |
| Outgoing weight col mean\|w\| at birth | ~0.080 |
| Early mean\|grad\| on weight row (grace) | **~7×10⁻⁷** pruned / **~3×10⁻⁶** survived |

**Verdict: RAW_NEAR_ZERO (confirmed).** Newborns’ true forward activations are near-zero vs established units; EMA warm-start only delays judgment while the EMA decays toward that true underactivation (age0 ema~0.7 → age1 ema~0.03). Extending grace alone will not help if the underlying activation stays dark — the prune mechanism is correctly reading a dead unit.

**Note on the grad figure:** ~1e−6 on new rows is **logged**, not yet a starvation proof — near-converged nets have small grads on useful established weights too. Matched baseline (init-origin early-life vs grown early-life at same age-since-init) is specified under the open follow-up step 2 in `phase1a_results.md`; do not over-interpret this row alone.

**Implication (at time of this probe):** next lever is **not** longer grace; whether new weights can learn is deferred to the ordered follow-up (trigger mistarget first, then matched-baseline grads).

### Dead ReLU vs attenuated-alive (mechanism split)

**Artifact:** `results/phase1a_diagnostics/newborn_dead_relu/summary.json`  
Full probe (~1024): pre-ReLU and post-ReLU per newborn at birth and age 1.

Definitions: `dead_relu` = ≥99% probe samples with post==0; `mostly_dead` = 50–99% post==0; `alive_attenuated` = <50% post==0 and mean post>0.

| Cohort | n | dead_relu | mostly_dead | alive_attenuated | mean frac post==0 | mean post |
|--------|--:|----------:|------------:|-----------------:|------------------:|----------:|
| All at **birth** | 61 | **44%** | 23% | 33% | 0.71 | 0.052 |
| Later **pruned** (at birth) | 37 | **68%** | 11% | 22% | 0.82 | 0.031 |
| Still **survived** (at birth) | 24 | **8%** | 42% | **50%** | 0.54 | 0.085 |
| All at **age 1** | 49 | **73%** | 0% | 27% | 0.74 | 0.012 |
| Later pruned (at age 1) | 37 | **81%** | 0% | 19% | 0.82 | 0.012 |
| Established init (ref) | 96 | **0%** | — | — | 0.34 | **1.27** |

**Verdict: MIXED, with dead-ReLU dominant among failures.**  
- At birth, **~67%** are dead or mostly dead; **~33%** are alive-but-small.  
- Units that eventually get pruned are **68–81% dead_relu**; survivors are rarely fully dead at birth (8%).  
- So both mechanisms exist: (1) dead-on-arrival ReLU is the **main** path into prune, (2) alive-attenuated is real but secondary.  
- Outgoing-weight scale alone cannot fix the dead majority; a positive birth bias / non-dead init for incoming pre-acts addresses (1); outgoing scale addresses (2) only.

**Fresh τ sweep (post-fix, stability metrics only):** all cells still fail thrash; ranking still relative (τ=0.5 lowest thrash/prune). **No absolute τ lock.**
| Variant B `μ` | `0.999` | No retune after primary |
| Prune edge fraction | `0.05` per layer per epoch | local z-score |
| Neuron prune | `z < -1` and contraction pressure | |
| Max new edges / layer / epoch | `64` | |
| Max new neurons / layer / epoch | `4` | |
| Seeds | `{0, 1, 2}` | |
| Stationary epochs | `20` | |
| Task-shift | B1 label perm; 15+15 | B3 optional |
| Train | SGD lr=0.01 momentum=0.9 wd=1e-4 batch=128 | Phase 0 match |

## Stability bar (concrete)

1. Final total hidden count ≤ `1.5 × (256+128+64)` unless accuracy Pareto-dominates fixed baseline.
2. Hard cap hit on ≤ **2** epochs per run.
3. Same-step free-then-reuse **must be zero successes** (hard cooldown); blocked attempts are logged as `cooldown_blocks`.
4. Cross-epoch thrash annotation if grow↔prune flip-flop on the same edge key within 3 consecutive epochs exceeds **N=10** events.

## Target-rate probe (required before τ sweep)

**Artifact:** `results/phase1a_diagnostics/target_rate_probe.json`  
**Protocol:** DynamicMLP Phase 0 init sizes, 1 full MNIST train epoch, `ActivationMagnitude` EMA (λ=0.99), seed 0. Active-only postsynaptic neuron scores.

| Layer | mean | median | p10 | p90 | frac > 0.1 | frac < 0.01 |
|-------|------|--------|-----|-----|------------|-------------|
| fc1 / h1 | **0.77** | 0.64 | 0.17 | 1.51 | 0.95 | 0.008 |
| fc2 / h2 | **1.02** | 0.93 | 0.03 | 2.05 | 0.86 | 0.055 |
| fc3 / h3 | **1.28** | 1.09 | 0.05 | 2.77 | 0.86 | 0.047 |

**Verdict on provisional r\*=0.1:** **miscalibrated (~8–10× too low).**  
With r\*=0.1 and τ=0.05, ≈84–91% of hidden units sit above r\*+τ (one-sided growth pressure).

**Verdict on single global r\*=0.9:** **also rejected.** Layer means differ by ~70% (h1 0.77 vs h3 1.28). A shared 0.9 puts most of h1 under target by construction (contraction bias) and h3 at/above target (growth bias) — measuring layer-scale artifact, not within-layer over/under-contribution. Same category of bug Phase 0 fixed with per-layer z-scoring before concatenated correlations.

**Locked r\*_ℓ for 1a (probe layer means, consistent statistic):**

| Layer | `r*_ℓ` (mean EMA) |
|-------|------------------:|
| h1 | **0.77** |
| h2 | **1.02** |
| h3 | **1.28** |

Median alternative (0.64 / 0.93 / 1.09) is recorded but not used; mean is the frozen choice. Do not retune after primary.

Raw batch mean |act| (end of epoch, no EMA) agrees in scale: h1≈0.75, h2≈0.99, h3≈1.25.

## Marginal growth pressure (A vs B — scope wording)

**Artifact:** `results/phase1a_diagnostics/growth_pressure_marginal.json`  
Protocol: 6 full epochs, seed 0, τ=0.25, strict eligibility.

| Epoch | n_growth / n_active | frac growth | frac contraction | GNC | candidates |
|------:|--------------------:|------------:|-----------------:|----:|-----------:|
| 0 | 141/448 | 0.315 | 0.406 | 0 | 0 |
| 1 | 121/384 | 0.315 | 0.276 | 0 | 0 |
| 2–5 | ~111–117 / ~282–341 | ~0.34–0.39 | ~0.07–0.19 | 0 | 0 |
| **mean** | **~119 / ~343** | **~0.35** | **~0.21** | **0** | **0** |

**Explanation B (locked for wording):** growth pressure is **not** rare — ~20–40% of active hidden units sit under growth pressure in every epoch. Local **growth+no_contraction conjunctions on missing edges** are what never appear at grow time (GNC=0). Not miscalibration of “too few growth-pressure neurons” (A). Follow-up for 1b: local activity correlation / geometry of pressure, not only r\*/τ scale.

## Variant A τ sweep (stability-bar re-read)

**Artifact:** `results/phase1a_diagnostics/tau_sweep_stability/summary.json`  
Protocol: seed 0, 6 full epochs, grid `{0.1, 0.2, 0.3, 0.5}`, r\*_ℓ locked, keep-strict.  
**Selection criterion (pre-registered):** stability bar — cap hits ≤2, final H ≤ 1.5× init, thrash events ≤10 within 3-epoch windows — **not** accuracy.

### Pre-registered metrics (primary)

| τ | cap_hit epochs | passes cap | final H (limit 672) | passes bound | neuron thrash | edge thrash | thrash total | passes thrash (≤10) | **stable** |
|--:|---------------:|:----------:|--------------------:|:------------:|--------------:|------------:|-------------:|:-------------------:|:----------:|
| 0.1 | 0 | yes | 267 | yes | 120 | 577 | 697 | **no** | **no** |
| 0.2 | 0 | yes | 267 | yes | 120 | 577 | 697 | **no** | **no** |
| 0.3 | 0 | yes | 284 | yes | 120 | 577 | 697 | **no** | **no** |
| 0.5 | 0 | yes | 332 | yes | 120 | 567 | 687 | **no** | **no** |

Accuracy (informational only): 0.981 / 0.981 / 0.982 / 0.985 — **not used to pick τ.**

**Absolute lock blocked:** every grid cell **fails** thrash ≤10 by a large margin. Cap and bound bars pass for all. Relative thrash is nearly flat (687–697); τ=0.5 is only marginally lowest.

**Soft-recommend (not final lock):** `τ=0.5` still ranks first on relative thrash + lowest neuron prune if a value must be chosen for primary; treat thrash bar as **known fail under current thrash definition + continuous edge prune + rate-capped slot reuse**, and monitor thrash on the 20-ep primary rather than claiming the absolute bar was met in the sweep.

### τ=0.1 vs 0.2 identical outcomes — wiring check

| Series (per epoch) | τ=0.1 | τ=0.2 |
|--------------------|-------|-------|
| `n_contraction_pressure` | 219, 158, 118, 91, 65, 53 | **198, 127, 80, 54, 32, 28** |
| `n_growth_pressure` | 173, 161, 152, 160, 149, 150 | **155, 131, 128, 126, 124, 121** |
| neuron prune counts | 76, 55, 41, 34, 24, 23 | **same** |
| hidden trajectory | identical to 0.2 | identical to 0.1 |

**Interpretation:** τ **does** reach `FixedThreshold` classification (contraction/growth series **differ**). Identical prune outcomes mean the intersection (contraction ∧ z < −1) is the same set of neurons in this run for both bands — not a silent wiring gap of the seed-path kind. Plausible, and now measured.

### Thrash identity + timing (before any thrash-bar decision)

**Artifact:** `results/phase1a_diagnostics/tau_thrash_detail/summary.json`

#### (1) Same count (120) — same IDs or different sets?

| τ | neuron thrash events | unique thrashing slot-IDs | Jaccard vs τ=0.1 |
|--:|---------------------:|--------------------------:|-----------------:|
| 0.1 | 120 | 42 | 1.0 |
| 0.2 | 120 | 42 | **1.0** (identical set) |
| 0.3 | 120 | 42 | **1.0** (identical set) |
| 0.5 | 120 | 41 | **0.89** (39 shared; 2 only-in-0.5, 3 only-in-0.1) |

All-four intersection: **39 / 42** of the τ=0.1 set.  
**Finding:** the constant **120 is not four independent equal-sized sets** — it is essentially **one shared pool of ~40 capacity slot-IDs** (lowest free indices under rate-capped grow + cooldown reuse), with only marginal τ=0.5 drift. Neuron thrash volume is **τ-insensitive** because it is driven by **rate-cap / slot-reuse mechanics**, not by homeostatic threshold width. Redefining the thrash bar to “make τ selection work” would mostly re-score a τ-orthogonal quantity for the neuron component.

#### (2) Front-loaded settle vs chronic thrash

Neuron thrash **events completing** by epoch (attribution = later epoch of the flip pair), τ=0.1–0.5 essentially identical:

| Completing epoch | Events |
|-----------------:|-------:|
| 1 | 24 |
| 2 | 24 |
| 3 | 24 |
| 4 | 24 |
| 5 | 24 |

- **20%** of events complete in epochs 0–1 (only ep1 can complete a flip; ep0 alone cannot).
- **80%** complete from epoch 2 onward — **flat**, not a one-time dense-start spike.
- First-thrash of *unique keys*: ~57% by ep1, rest drip in through ep2–4 — keys often appear early, but **event rate does not drop**.

**Finding:** thrash is **chronic / steady-state** under this cadence and rate cap, not solely dense-start shakeout. A windowed bar that only drops epochs 0–1 would still leave ~96 events (≫10) from ep2–5. Redefining the bar to ignore early epochs is **not** justified as a settle correction by this series; that would be a post-hoc rescue of the absolute fail.

### Newborn cold-prune mechanism (measured)

**Artifact:** `results/phase1a_diagnostics/newborn_cold_prune/summary.json`  
Hypothesis: a newly grown slot starts with activation EMA ≈ 0, sits far below `r*_ℓ`, registers as contraction (+ low z), is pruned on the **next** plasticity step; the same low free index is reclaimed by rate-capped growth; cycle.

| Quantity (τ=0.5, 6 ep; τ=0.1 identical shape) | Value |
|-----------------------------------------------|------:|
| All grown-slot birth→prune events | 60 |
| **lag = 1 (next plasticity step)** | **60 / 60 (100%)** |
| lag = 0 or ≥2 for birth→prune | 0 |
| Thrash grow→prune events | 60 |
| **grow→prune thrash lag = 1** | **60 / 60 (100%)** |
| Thrash prune→grow (slot freed then reclaimed) | 60 (lag 1: 85%, lag 2: 15%) |
| Unique keys in grow→prune thrash | ~40–42 |

**Verdict: CONFIRMED.** Every grown neuron that is later pruned in this window is pruned on the **immediately following** plasticity step. That matches newborn cold-start vs `r*_ℓ`, explains flat/chronic/τ-orthogonal thrash, and explains the shared low-index slot pool (deterministic free-slot selection after cooldown).

**Implication for thrash bar / τ lock:** neuron thrash is a **newborn-grace / signal-warmup** issue, not a τ-selection issue. Softening thrash N or dropping early epochs does not address the mechanism. Named fixes (if desired later): grace period before a grown slot is eligible for homeostatic prune; seed EMA at `r*_ℓ`; or warm-up steps of signal before prune eligibility — each a deliberate rule change, not a post-hoc bar tweak.

### Open before absolute τ lock (options, not chosen)

Data rules out “redefine thrash by dropping early epochs only” as a settle-based fix, and shows neuron thrash is largely **τ-orthogonal**. Remaining honest paths:

1. **Longer primary-shaped run** (full 20 ep) with thrash time series — confirm chronic thrash continues or damps after epoch 6 (current data stops at 6).  
2. **Named rule change** that targets rate-cap / slot-reuse thrash (not a softer N).  
3. **Named thrash metric change** that excludes rate-cap slot cycling if that is declared out of scope for the bar — requires explicit prereg amendment, not “N=10 but after epoch 2.”  
4. Soft-run a τ only with thrash fail documented as **known fail of the bar under current mechanics**, not as pass.

**Do not lock τ=0.5 as stability-bar pass.** Soft-recommend remains relative only.

## Growth-fire gate (required before locking τ from the sweep)

Short smokes and miscalibrated r\* only proved **prune** is live. Growth has not been shown under correctly scaled per-layer targets. **Do not lock τ from the short sweep alone.**

**Gate protocol:**
1. Run short Variant-A τ sweep (seed 0) with locked `r*_ℓ` and candidate τ grid.
2. Take the leading τ by stability (bounded counts, low thrash, prune-by-origin sensible).
3. **Before locking:** re-run that τ alone, seed 0, **5–10 epochs** (full steps/epoch, not 5-step smoke), same `r*_ℓ`.
4. **Pass** if cumulative `edges_grown + neurons_grown ≥ 5` (handful) over the window and at least one plasticity step has `edges_grown > 0` or `neurons_grown > 0`.
5. **Fail** if growth never fires: treat incentive threshold / eligibility as too conservative; adjust (named change: e.g. lower edge bar, relax eligibility, or raise max_new) **before** primary multi-seed — do not discover this after a 20-epoch × 3-seed commitment.

Artifact path for the gate run: `results/phase1a_diagnostics/growth_fire_gate/`

**Gate result (2026-07-21, seed 0, 6 full epochs, r\*_ℓ locked as above):**

| τ | edges grown | neurons grown | steps w/ growth | prune edges | gate |
|--:|------------:|--------------:|----------------:|------------:|:----:|
| 0.15 | 429 | 72 | 6/6 | 117 806 | **PASS** |
| 0.25 | 429 | 72 | 6/6 | 115 265 | **PASS** |
| 0.40 | 429 | 72 | 6/6 | 103 675 | **PASS** |

Growth is live under correctly scaled per-layer targets (not only under the rejected global r\*=0.1). In this window growth counts hit the per-epoch rate caps (≤4 neurons/layer/epoch × 3 × 6 = 72), so τ mainly differentiated **prune volume** (higher τ → less contraction pressure → fewer prunes) rather than growth count — the short stability sweep should still use that prune-side signal, with growth confirmed alive before primary.

Train acc end of gate runs ≈ 0.98 (plastic net still learns).

### Pre-cap / prune-path wiring check (required read before τ sweep)

**Artifact:** `results/phase1a_diagnostics/growth_fire_gate/precap_wiring_check.json`  
Logs now split: `edge_grow_precap` / `neuron_grow_precap`, `edges_grown_incentive` vs `edges_grown_seed`, `edges_pruned_competitive` vs `edges_pruned_collateral`.

**Q1 — Why post-cap growth was identical across τ?**

| Metric (6-ep sum) | τ=0.15 | τ=0.25 | τ=0.40 |
|-------------------|-------:|-------:|-------:|
| **neuron_grow_precap** | **869** | **716** | **547** |
| neuron grown (post-cap) | 72 | 72 | 72 |
| **edge_grow_precap (incentive+τ path)** | **0** | **0** | **0** |
| edges_grown_incentive | 0 | 0 | 0 |
| edges_grown_seed (birth wiring) | 429 | 429 | 429 |

- **Neuron growth:** pre-cap **falls with τ** (869→547) while post-cap stays 72 → **true rate-cap saturation**. τ **is** wired into growth pressure; the sweep’s neuron-side bar is real.
- **Edge growth:** incentive-path pre-cap is **0 for all three τ**. Every grown edge is from **`_seed_edges_for_neuron`**, which currently accepts `incentive > 0` and **does not apply τ**. Identical edge totals are therefore **not** evidence that edge-τ is working — seed path bypasses τ and is driven 1:1 by neuron births (themselves cap-saturated).
- **Named issue before primary:** either (a) apply the same edge threshold to the seed path, or (b) document seed edges as τ-free structural bootstrap and treat Variant A’s edge-τ as only the incentive path (which is currently silent under dense-start). Do not claim the τ sweep tunes edge formation while seed path dominates.

**Q2 — “Fewer prunes at higher τ” — which path?**

| Metric (6-ep sum) | τ=0.15 | τ=0.25 | τ=0.40 |
|-------------------|-------:|-------:|-------:|
| **neurons_pruned** | 253 | 247 | 215 |
| **edges_pruned_collateral** (with neuron death) | 64 723 | 62 125 | 49 202 |
| **edges_pruned_competitive** (z-score / 5%, no τ) | 53 083 | 53 140 | 54 473 |

- Neuron prunes and **collateral** edge removals fall with τ — expected (wider neutral band → less contraction pressure).
- Competitive edge prune is ~flat / slightly up (graph-size + fixed 5% fraction); it does **not** read τ. Earlier “edge prune moves with τ” in the unsplit total was almost entirely **collateral**, not a second τ wire into competitive scoring.

**Sweep implication:** short τ sweep is meaningful for **homeostatic pressure width** (neuron grow pre-cap, neuron prune, collateral). It is **not** yet a test of incentive-based edge formation under τ. Resolve seed-vs-τ policy (named) before treating edge-length / grown-edge diagnostics as τ-sensitive.

### Standalone edge path: opportunity vs threshold (pre (a)/(b) decision)

**Artifact:** `results/phase1a_diagnostics/growth_fire_gate/edge_opportunity_check.json`  
Counters (summed over 6 epochs, seed 0):

| Metric | τ=0.15 | τ=0.25 | τ=0.40 |
|--------|-------:|-------:|-------:|
| `missing_in_radius` (absent edges, incentive>0 ⇒ d≤R) | 4 343 | 4 890 | 11 356 |
| `candidates_evaluated` (eligibility-pass; pre-τ) | **0** | **0** | **1** |
| `threshold_passes` / `edge_grow_precap` | 0 | 0 | 0 |
| grown via incentive path | 0 | 0 | 0 |

**Per-epoch pattern (all three τ similar early):** epochs 0–1 have `missing_in_radius=0` despite large competitive prune counts; local gaps only accumulate from epoch 2 onward (hundreds → thousands).

**Interpretation (neither pure “broken τ” nor pure “dense-start still fully wired”):**

1. **Early window:** competitive prune removes many edges, but almost none leave a **local** (d≤R, incentive>0) hole among surviving actives — consistent with dense-start wiring many **long-range** edges (coords random/grid; most pairs have d>R=0.35). Those gaps are invisible to the incentive path by design of R.
2. **Later window:** local gaps **do** appear (`missing_in_radius` ≫ 0), so “no opportunity ever” is false for the full 6 epochs.
3. **Eligibility, not τ, is the bottleneck on those gaps:** `candidates_evaluated ≈ 0` (one pair total at τ=0.40). Nearly every local gap fails the growth-pressure / no-contraction gate. τ is almost never applied to a real candidate; `threshold_passes=0` is **not** evidence that τ is mis-set, because the threshold stage is starved.

**Implication for (a) vs (b):** do not frame 0 incentive-grown edges as “seed is a deliberate τ-free bootstrap” alone, and do not frame it as “incentive formula/τ broken” alone. Accurate statement: *under dense-start + R locality, the standalone path only sees pruning-created local gaps; in this gate window those gaps almost never clear eligibility, so edge-τ is untested; observed edge growth is seed-only.* Choosing (a) or (b) should follow from whether 1a wants the seed path to carry structural edge birth under τ, or wants a longer / different regime that produces eligibility-passing refill opportunities before claiming edge-τ is live.

### Eligibility-gate A/B (before seed/τ decision)

**Config flag:** `plasticity.edge_eligibility_mode`  
- `neither_contraction` — strict plan default (≥1 growth, **0** contraction)  
- `not_both_contraction` — relaxed probe (≥1 growth, not **both** endpoints contracting)

**Artifact:** `results/phase1a_diagnostics/eligibility_gate_ab/`  
Protocol: 6 full epochs, seed 0, r\*_ℓ locked, τ=0.25, fixed threshold.

| Mode | missing_in_radius | candidates_evaluated | threshold_passes | grown_incentive | grown_seed |
|------|------------------:|---------------------:|-----------------:|----------------:|-----------:|
| strict `neither_contraction` | 4890 | **0** | 0 | 0 | 429 |
| relaxed `not_both_contraction` | 4890 | **0** | 0 | 0 | 429 |

**Result: relaxing the contraction veto does not move `candidates_evaluated` off zero.** Bit-identical structure totals. The hypothesized “growth partner blocked by one contracting endpoint” pool is empty in this regime.

**Why (gap pressure taxonomy, post-step local gaps):**  
`results/phase1a_diagnostics/eligibility_gate_ab/gap_pressure_taxonomy.json`

| Pattern among missing-in-radius pairs (sum over epochs) | Count |
|----------------------------------------------------------|------:|
| **all_neutral** (no growth, no contraction on hidden ends) | **9675** |
| no_growth_some_contraction | 521 |
| growth_no_contraction (would pass **strict**) | 536 |
| growth_plus_partial_contraction (would pass **relaxed only**) | **0** |

So:
1. **Dominant failure mode is not contraction veto — it is absence of growth pressure** on gap endpoints (neutral holes left by prune).
2. Relaxed mode’s exclusive bucket is **empty** → null A/B outcome is expected, not a bug in the flag.
3. Some strict-pass pairs exist post-step (~536) but `candidates_evaluated=0` during grow → consistent with **same-step cooldown** (edges pruned this plasticity step cannot be refilled until the next) and/or pressure/geometry timing; they are not unlocked by loosening contraction.

**Honest scope options (data-backed, still open product choice):**

- **Keep strict gate + seed-dominated edges:** report explicitly that *edge formation in 1a is dominated by birth-seed bootstrap; the standalone incentive path sees local gaps but almost never eligibility-passing growth-side partners at this horizon — distance-cost/incentive wiring is only weakly exercised.* Do not let seed-grown edges stand in for “incentive-driven wiring” in the write-up.
- **Change a different lever than contraction veto** if 1a must exercise standalone refill: e.g. allow high-incentive **neutral** endpoints, defer refill to next epoch with logged cooldown stats, longer horizon, or sparse init — each is a **named** rule change; the one-line contraction relaxation is **ruled out** as insufficient by this A/B.

Default config remains `neither_contraction` until an explicit choice is recorded.

### Persistence trace + same-function check (supersedes the “536 GNC” figure)

**Artifact:** `results/phase1a_diagnostics/eligibility_persistence/persistence_trace.json`

**Same-function check:** the orchestrator imports and calls
`neuronca.plasticity.rules.select_edges_to_grow` by identity (`is`). Pair
enumeration for growth is **not** a second walk. In-path stats
(`growth_no_contraction`, cooldown splits, `candidates_evaluated`, pair IDs)
are collected **inside that same loop** when `collect_edge_pairs=True`.

**Withdrawn:** the earlier post-step `classify_gaps` taxonomy that reported
~536 `growth_no_contraction` pairs. That was a **parallel, post-step**
implementation (no cooldown, different timing). It is **not** what the growth
mechanism sees and must not be used to argue for a cooldown-only story or a
candidate-generation bug.

**Live-path totals (6 epochs, τ=0.25, seed 0, same run as training):**

| Metric (sum over grow phases) | Value |
|-------------------------------|------:|
| missing_in_radius | grows by epoch (0 → 2464 at ep5) |
| **growth_no_contraction** (strict pattern, same loop) | **0** |
| gnc on cooldown | 0 |
| gnc off cooldown | 0 |
| **candidates_evaluated** | **0** |
| blocked_cooldown | dominant on early local gaps |
| blocked_eligibility | rises later (ep4–5) |
| pairs that persist as gnc from t→t+1 | **0** (no gnc pool to persist) |
| bug: persist + off-cd + not candidate | **0** |

Per-step: `gnc_off_equals_candidates` is always true (both empty sets).

**Verdict (measured, not hypothesized):**

1. **Not a candidate-generation bug.** There is no live-path pool of
   growth+no_contraction pairs that the enumerator fails to promote; the
   strict GNC count at grow time is zero every epoch.
2. **Not “536 opportunities stuck on one-step cooldown.”** That number was
   from the invalid parallel taxonomy. On the live path, GNC never appears
   at the grow phase at all.
3. **What the live path actually sees:** local gaps (`missing_in_radius`)
   open over time; at grow time they are almost entirely
   **cooldown-blocked** (just pruned this step) and/or
   **eligibility-blocked without being GNC** (neutral / contraction-without-growth
   endpoints). Cumulative `candidates_evaluated` remains 0.
4. **Persistence:** with no GNC pairs at t, there is no t→t+1 GNC survival
   curve to report. The precise statement is: *under once-per-epoch cadence,
   the grow phase never observes a strict growth+no_contraction local gap;
   the earlier 536 figure was diagnostic contamination from a separate walk.*

**Scope implication (updated):** keep-strict remains defensible; the honest
write-up should say that **standalone incentive candidacy never fired in
this window on the real growth path**, not that “536 GNC pairs were blocked
by cooldown.” Seed still dominates all edge birth. Cooldown is real for
fresh prunes (`blocked_cooldown` ≫ 0) but does **not** explain a phantom
GNC/candidates gap.

## Smoke structure liveness (required before τ sweep)

**Artifact (original CLI smoke):** `results/phase1a_smoke_fixed/stationary/seed_0/`  
Config: r\*=0.1 (wrong), τ=0.05, 1 epoch × 5 train steps.

| Event | Count |
|-------|------:|
| edges pruned (`init_dense`) | **35 763** |
| edges pruned (`incentive_grown`) | 0 |
| edges grown | **0** |
| neurons pruned | **50** (h1 −34, h2 −16, h3 0) |
| neurons grown | **0** |
| cooldown blocks (same-step free-reuse denied) | **46** |
| final `incentive_grown` edges | **0** |

**Verdict:** mechanism is **not static** — competitive/homeostatic **pruning is live**, cooldown guard **fires**. Under miscalibrated r\* and a 5-step smoke budget, **growth did not fire** (all remaining edges still `init_dense`). Density among survivors stayed ~0.95–1.0 (prune-from-dense).

**Follow-up liveness (20 steps × 2 epochs):** `results/phase1a_diagnostics/structure_liveness.json`

| Setting | edge prune | edge grow | neuron prune | neuron grow |
|---------|:----------:|:---------:|:------------:|:-----------:|
| r\*=0.1, τ=0.05 (miscalibrated) | yes | yes (92) | yes | yes (net still shrinks) |
| r\*=0.9, τ=0.2 (probe-scale) | yes | no (in this short window) | yes | no |
| r\*=1.0, τ=0.3 | yes | no | yes | no |

Dense-start scope limitation stands: early dynamics are prune-dominated; growth needs either longer runs, hotter local pockets, or explicit under-dense init (out of scope for 1a primary). Sweep should still use probe-locked r\* so homeostatic **direction** is not globally one-sided.

## Diagnostics (required, both variants)

- Stability (counts over time, cap hits, thrash)
- Sparsity (vs dense active baseline)
- **Prune-by-origin** (init_dense vs incentive_grown rates and survival %)
- Pareto (acc vs active param count) vs fixed-topology baseline — **§Pareto comparison (pre-registered)**
- Continual adaptation (B1; structure event rate around shift)
- Connection-length histogram: **all / init_dense / incentive_grown**

## Pareto comparison (pre-registered before baseline runs)

**Status:** **LOCKED before fixed-topology comparison runs.** Machine-readable copy:  
`results/phase1a_diagnostics/pareto_comparison_prereg.json`.  
Do not retune δ or the size grid after seeing baseline results.

### Horizon caveat (same class as the 79% stays number)

The **20-epoch accuracy snapshot is mid-trajectory, not long-run equilibrium.** Early-cohort still-stays is still declining at age 18 under fixed (~0.95→0.72) and reduced-but-not-zero under sliding (~0.95→0.83). Specialist contribution to accuracy may still be moving at the Pareto snapshot.

**Write-up rule:** report Pareto as **efficiency at 20 epochs, alongside evidence that the specialist population is still mid-decline** (cite cohort curves). Do **not** present epoch-20 Pareto as “how efficient this architecture is,” full stop.

**Does not block the comparison:** run the 20-ep Pareto **now** on **both** plastic variants with this caveat stated. A 40–60 ep confirmation is a **named follow-up**, not a gate (same pattern as 6-ep mechanism → 20-ep hold).

### What “matched final parameter count” means

It is **not** “pick a size up front and train both arms at it.” Plastic final size is **endogenous**.

It **is** a **post-hoc Pareto** comparison:

1. Run plastic recipes first (done: **fixed and sliding** 20-ep holds, seeds {0,1,2}).
2. Record **active parameter count** and **test accuracy** per seed **per arm**.
3. Train **fixed-topology dense** MLPs once on a **pre-declared size grid** spanning the observed active-param masses and H.
4. Plot both plastic arms against the **same** baseline grid; score each arm with its own \(P_p, A_p\).

### Both plastic arms are required (not optional)

Sliding is **not** a side series to defer. Fixed early-cohort erosion (~22 pp still-stays drop age 5→18) is cut to ~12 pp under sliding (`B_REDUCES_EARLY_EROSION`). If that matters for the **headline** claim (task performance per parameter), it must show up on the Pareto plot — not only in specialization diagnostics. 6-ep `B_NO_HELP` never tested this.

**Plastic reference — fixed stack (20-ep hold):**

| Seed | H sum | Active params | Test acc |
|-----:|------:|--------------:|---------:|
| 0 | 562 | 88 707 | 0.9665 |
| 1 | 468 | 88 458 | 0.9742 |
| 2 | 475 | 88 492 | 0.9564 |
| **Mean** | **502** | **88 552** | **0.9657** |

Acc seed range **1.78 pp**, sample std ≈ **0.73 pp**.

**Plastic reference — sliding stack (20-ep hold):**

| Seed | H sum | Active params | Test acc |
|-----:|------:|--------------:|---------:|
| 0 | 511 | 85 165 | 0.9579 |
| 1 | 425 | 83 749 | 0.9721 |
| 2 | 414 | 83 422 | 0.9569 |
| **Mean** | **450** | **84 112** | **0.9623** |

Acc seed range **1.52 pp**. Mean active params **~3.4k lower** than fixed.

**Parameter axis:** plastic = `active_parameter_count`; fixed-topo = full dense weight+bias at declared H. Param-matched dense (~80–110k) and H-matched dense (~243–324k) are different points — both required.

### Fixed-topology size grid (locked; shared by both arms)

Train recipe identical to plastic except no structure changes: SGD 0.01 / mom 0.9 / wd 1e−4 / batch 128 / **20 ep** / seeds {0,1,2}. Base: `configs/phase1a_baseline_fixed_topo.yaml` + `model.hidden` override.

| Tag | H | ≈ dense params | Role |
|-----|---|----------------:|------|
| `tiny` | [64, 32, 16] | 53 018 | below both plastic P |
| `param_lo` | [96, 48, 24] | 81 442 | brackets sliding P; just below fixed P |
| `param_match` | [112, 56, 28] | 96 134 | nearest dense ≥ fixed plastic P |
| `param_hi` | [128, 64, 32] | 111 146 | above both plastic P |
| `phase0_init` | [256, 128, 64] | 242 762 | Phase 0 / plastic init size |
| `H_mid` | [256, 128, 86] | 245 820 | dense at mid plastic H |
| `H_hi` | [336, 140, 88] | 324 238 | dense at upper plastic H |

For **each** plastic arm, interpolate \(A_b\) linearly in **log(params)** between the two cells that bracket **that arm’s** mean active_params. Do not extrapolate off-grid.

### Numeric pass/fail bar (locked)

**δ = 1.0 percentage point** on **test** accuracy (absolute 0.01).

**Justification:** fixed 20-ep seed range **1.78 pp** (half-range ≈ 0.9); sliding range **1.52 pp**. Round to **1.0 pp**. τ-sweep single-seed diffs are not multi-seed noise — not used for δ.

**δ uncertainty (n=3 — state before results):**  
δ is the right operational choice given available data, but a range/std from **only three seeds is itself a noisy estimate** of true seed variance. **If a verdict lands within ~0.5 pp of the boundary**, annotate **`NEAR_BOUNDARY_DELTA_UNCERTAIN`** — do not treat a narrow pass as a clean win or a near-miss as a clean fail. **Do not change δ after seeing results**; only flag confidence.

**Primary predicates (both required):**

| ID | Arm | PASS if |
|----|-----|---------|
| `PARETO_ACC_AT_COST_FIXED` | fixed plastic | \(A_p^\mathrm{fix} \ge A_b(P_p^\mathrm{fix}) - 0.01\) |
| `PARETO_ACC_AT_COST_SLIDING` | sliding plastic | \(A_p^\mathrm{slide} \ge A_b(P_p^\mathrm{slide}) - 0.01\) |

**Cross-arm annotation (not a dense-baseline pass/fail):**  
`SLIDING_VS_FIXED_ON_PARETO` — whether sliding’s \((P,A)\) dominates fixed, is dominated, or is a trade-off. This is the direct test of whether reduced erosion improves **task-performance-per-parameter**.

**Also report:** per-seed points; `DOMINATED_BY_SMALLER_DENSE` per arm; cells with dense params \(\ge P_p\).

**Do not:** retune δ; omit sliding; drop the mid-decline caveat; block on a 40–60 ep run.

### Protocol order

1. Plastic 20-ep holds complete for **fixed and sliding** (sizes/acc frozen above).  
2. **This section + `pareto_comparison_prereg.json` published** ← you are here.  
3. Run all fixed-topo grid cells **once** (shared baselines).  
4. Score both predicates; annotate sliding-vs-fixed and any near-boundary uncertainty.  
5. Named follow-up (not blocking): 40–60 ep plastic holds if a long-run equilibrium claim is needed.

## Non-goals

No relay, scaffolding, specialization metrics, topology optimizer, skip connections, post-primary retune of τ/μ, redefinition of `activation`.
