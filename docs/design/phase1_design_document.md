# Phase 1: Structural Plasticity Design Document

## Status

Phase 1 begins where Phase 0 stopped: Phase 0 validated that a locally-computable
signal (`activation` magnitude, confirmed robust across shift type, convergence
state, and ground-truth measure — see `phase0_consolidated_results.md`) can proxy
for gradient-based importance without an orchestrator ever seeing a gradient
directly. Phase 1 builds the orchestrator itself: an NCA-governed mechanism that
grows, prunes, and wires a reasoning network's structure using only that local
signal and local rules — no global view, no gradient access, no centralized
architecture search.

This document sets objectives and phasing before any implementation spec is
written. Each phase below gets its own separate, pre-registered implementation
spec when it's time to build it — this document is the roadmap, not the build
instructions.

---

## 1. Objective

Determine whether a reasoning network's structure — which connections exist,
how many neurons are allocated, which regions specialize — can be shaped
entirely by local rules and local information, driven by pressure (resource
scarcity, connection cost, task demand) rather than by hand-specifying what
structure should exist.

This is a direct continuation of the reframe that motivated the whole project:
capabilities should emerge because an environment makes them useful, not
because they're encoded. PSMCL's original failure mode — deciding what a
self-model needs to contain and building components to produce exactly that —
is the same failure mode Phase 1 has to avoid at the architecture level:
deciding what structure the network needs and hand-wiring it in, rather than
creating conditions under which that structure becomes the cheapest way to
solve the task.

**What "success" means for Phase 1 as a whole:** a reasoning network whose
structure changes over training — through growth and pruning driven by local
signals — matches or outperforms an equivalent fixed-topology network on
task performance per unit of resource (parameters, connections), and does so
via rules that never require centralized information. Secondary, weaker
success: even without outperforming a fixed baseline, observing structure
that responds sensibly to task pressure (e.g., growth concentrating where a
task genuinely needs more capacity, pruning concentrating where it doesn't)
would be informative on its own.

---

## 2. Core design commitments (carried forward, not re-litigated here)

These were established through extended discussion earlier in this project and
are treated as fixed constraints for all of Phase 1, not open questions:

- **The reasoning network trains via ordinary backpropagation, full stop.**
  The orchestrator never has gradient access. This was the design correction
  that also explained GEVB-RF's Phase 0 failure — feedback alignment's
  rotation mechanism requires FA-driven weight updates, which this design
  deliberately never uses. Do not revisit this for Phase 1 without treating
  it as a real architecture change, not a tweak.
- **The orchestrator's information channel is local activity plus a broadcast
  signal** — `activation` magnitude (Phase 0's validated primary) as the
  default importance signal, with `eligibility_residual_energy` retained as
  the one to watch for shift-recovery dynamics specifically, since it
  recovers genuinely faster than `activation` once the frozen-weight artifact
  in `activation`'s apparent "instant recovery" is accounted for.
- **Growth and pruning must be homeostatic, not hotspot-seeking.** A
  unidirectional "more activity → more resources" rule is a preferential-
  attachment dynamic with no natural ceiling; the biological and computational
  precedent (below) both use bidirectional deviation-from-target rules
  instead. Pruning must be continuous and competitive, not a one-time event
  triggered only when storage runs out.
- **No cell has global view.** Any mechanism that lets a cell directly
  address or reference a distant cell's coordinates (rather than discovering
  it through local propagation or relay) breaks this constraint and should be
  treated as a design compromise to flag explicitly, not a detail.
- **Structural decisions are governed by a local plasticity rule, not by
  credit assignment against task loss — resolved, not an open question.**
  Early framing treated discrete growth/pruning decisions as a harder version
  of the continuous-weight credit-assignment problem (needing either a
  differentiable relaxation or an evolutionary/RL outer loop, as in the
  HyperNCA precedent). That framing is superseded. The adopted design instead
  asks "can pruning itself be a local learning rule?" — each cell decides
  locally (consistently overactive, consistently silent, correlated with
  neighbors) using homeostatic and Hebbian-style local rules, with **no
  optimizer, gradient, or reward signal over topology at all**. This gives up
  the guarantee that structural updates optimize the same objective as the
  weights; Phase 0's finding that `activation` — a signal with no task
  awareness — already correlates well with task-gradient importance is the
  concrete reason to expect this gap matters less in practice than it would
  in general. Success is measured against stability, sparsity, resource
  efficiency, continual adaptation, and (optionally) emergent specialization
  — not against exact optimization of task loss by the structural process
  itself.
- **The plasticity rule's own operating point may adapt locally and
  continuously, without this becoming task-level optimization.** Biology's
  actual mechanism here is metaplasticity — the plasticity rule's threshold
  (e.g., BCM's sliding threshold, where each neuron's own time-averaged firing
  rate sets the boundary between strengthening and weakening) drifts based on
  a cell's own activity history, with the specific, established function of
  stabilizing otherwise-unstable local plasticity. This is a **third**
  option, distinct from both "fixed forever" and "learned against task loss":
  the set-point adapts, but only from local, non-task information. Phase 1a
  tests this directly as a design variant (see below) rather than assuming a
  fixed constant threshold is the only local-only option.

---

## 3. Phasing

Staged deliberately, one mechanism per stage, mirroring the single-variable
discipline that made every Phase 0 result attributable rather than confounded
(most concretely demonstrated by the RNG-leakage bug hunt, which only
resolved because each test changed exactly one thing at a time). Each stage
reuses the prior stage's code unmodified and adds exactly one new mechanism.
No stage's implementation spec gets written until the prior stage's result is
in.

### Phase 1a — Baseline structural plasticity

> **Status (diagnostic freeze — read before planning off this section):**  
> Under **kaiming-random** birth (freeze default), **grown neurons do not
> permanently establish** (grace-buffered revolving door; final hidden size
> from residual dense-init prune, not successful refill). Full chain in
> [`docs/phase1a_results.md`](../phase1a_results.md).  
> **Real design gap (not “needs more grace”):** biological newborns inherit
> **structured, task-relevant first inputs by placement** (e.g. perforant
> path into new dentate granule cells); freeze growth **decouples** trigger
> (“this region is stressed”) from birth (blind Kaiming noise). Critical
> period ≠ longer adult rule — immature cells use a **more permissive
> learning rule** for a window. **Parent-perturbed birth establishes but
> clones**. **Primary 1a result (6 ep):** `gap_imprinted` stack under fixed
> and sliding (seeds 0–2): ~53% establish, **~91–92% stays specialized**;
> sliding ≈ fixed (`B_NO_HELP` — within-arm seed spread ≫ between-arm Δ).
> Early τ-sweep A/B **superseded** as headline. **Open:** 20-epoch hold.
> Product defaults not yet flipped.  
> **Do not plan Phase 1b/1c** as if 1a already runs a working grow/prune
> loop under random birth. Relay/scaffolding inherit the same risk unless
> new structure is **born into** information-carrying pathways, not noise.

**What it adds:** homeostatic growth (deviation from a target firing rate
drives expansion or contraction, not unidirectional hotspot-seeking) and
continuous competitive pruning, both driven by the `activation` signal, with
a naive distance-cost mechanism: an incentive signal that propagates through
the grid one cell per timestep, decaying with hops traveled, so distance cost
falls out of the propagation physics rather than being separately computed.
No relay, no scaffolding reuse — those are 1b and 1c.

**Why this is first:** it's the smallest version of the whole idea — pressure-
driven growth/pruning without any of the distance-specific refinements — and
it's the necessary baseline every later stage gets compared against. Without
it, 1b and 1c have nothing to show improvement over. *(Baseline value after
the freeze is still real: measured non-establishment and prune-dominated
dynamics are the comparison substrate, not an assumed healthy refill loop.)*

**Known limitation going in, not to be treated as a surprise later:** the
propagation-decay distance signal has an unresolved false-negative problem —
a cell with genuine incentive for a long-range connection is indistinguishable
from one with no incentive, if the signal simply hasn't propagated far enough
yet. This is expected to show up as an exploration bottleneck (few or no
long-range connections forming) and is the specific problem 1b exists to fix.
Track connection-length distribution from the start so this is measured, not
assumed. **Additional measured limitation (post-implementation diagnostics,
not in the original design list):** standalone incentive edge formation is
inert in the dense-start / keep-strict window; observed edge birth is seed-
bootstrap. Separately, **neuron growth does not establish** — see status box.

**Design variant to test within 1a, not deferred to a later stage:** fixed
homeostatic threshold (a single hand-set constant, tuned once via manual
sweep, same cadence as the GEVB LR/scale diagnostic) versus a metaplastic
sliding threshold (each cell's own growth/pruning set-point drifts based on
its own recent activity history, BCM-style — still local, still no task
objective, just a second, slower local feedback loop on top of the first).
**Original framing (preserved as intent):** this was the central open
question once credit assignment is off the table — does a self-adjusting
local set-point produce more stable dynamics than a hand-picked constant
one, under a live, coupled feedback loop where growth changes activity,
which changes the growth signal, which changes activity again.  
**Current framing (superseding for empirics):** that A/B only measures a
meaningful difference in *growth quality* if growth can establish under
either variant. In the instrumented window it does not; thrash / final-H /
stability-bar numbers primarily reflect **prune dynamics on dense-init**
plus a grace-length conveyor of unestablished newborns. Treat fixed-vs-
sliding as still implementable and reportable side-by-side, but **not** as
the headline 1a scientific claim until establishment is fixed or explicitly
de-scoped. Soft `τ` ranking exists; absolute `τ` lock and primary multi-
seed are **held** (see results doc).

**Named diagnostics, required for both variants, not optional
nice-to-haves:**
- **Stability:** bounded parameter/connection count over a trailing window
  (no sustained monotonic growth) and absence of oscillation (repeated
  grow/prune flip-flop on the same region). *Read with freeze: thrash bar
  fails the whole pre-primary grid; mechanism is non-establishment, not
  “wrong τ.”*
- **Sparsity:** connection density relative to a fully-connected baseline of
  the same neuron count. *Tag init-origin vs grown-origin; final H ≠ successful growth.*
- **Energy efficiency:** the Pareto criterion below.
- **Continual adaptation:** reuse Phase 0's task-shift lag-curve methodology
  (including the `t_star_sustained` diagnostic, not just first-crossing)
  applied to the plastic topology, not just to a fixed one.
- **Establishment (added by freeze):** fraction of grown neurons that remain
  active past grace / first judgment; weight-driven vs scaffold post-activation
  for that cohort. Zero establishment is a first-class result, not a
  precondition bug to hide before reporting other metrics.

**Success criterion (to be finalized before implementation, not after):**
task performance per parameter count, plotted as a Pareto curve against a
fixed-topology baseline of matched final size — the same shape of comparison
used in the growth-vs-pruning ablation literature that showed unconstrained
growth reaching similar performance at two orders of magnitude more
parameters than growth-with-pruning. A pass here means the pressure-driven
network reaches comparable performance at comparable or lower parameter cost;
a fail (ballooning parameter count for no gain, or persistent thrashing
between growth and pruning) is a real, reportable result, not a bug to
iterate away before reporting. Report the fixed-threshold and
sliding-threshold variants side by side on all four diagnostics above — do
not pick one to report and discard the other.  
**Freeze pointer:** those criteria assumed a functioning grow/prune loop.
The pre-primary diagnostic freeze instead reports **growth does not
establish** as the headline 1a finding; Pareto / stability / sparsity /
continual-adaptation numbers, when eventually filled from primary, must be
interpreted under that constraint (and primary itself is held until the
named follow-up or an explicit growth-as-cosmetic decision). See
`docs/phase1a_results.md`.

### Phase 1b — Relay / landmark decomposition

**What it adds:** intermediate "waypoint" cells. Instead of one incentive
signal propagating the full distance between two cells, growth proceeds as a
chain of short, independently-solvable local hops — each hop a full local
decision, not a long-distance judgment made from decayed signal strength.

**Why this is second, not first:** it's a direct, targeted fix to 1a's known
limitation, and comparing it against 1a is only meaningful once 1a's baseline
behavior (including the false-negative problem) is actually measured, not
assumed from design intuition.

**Shared risk from 1a freeze (do not ignore when specifying 1b):** new
structure (relay waypoints, new hops) inserted into a near-converged network
can inherit the same **non-establishment / weak gradient-pressure** failure
as 1a neuron growth. 1b success criteria should include an establishment-
style check on new relay units/edges, not only connection-length histograms.
See `docs/phase1a_results.md` §Implications for Phase 1b/1c.

**Success criterion:** long-range connections form that would not have formed
under 1a's propagation-decay alone (or form measurably faster/more reliably).
If 1b produces no measurable difference from 1a in connection-length
distribution or formation reliability, that's informative — it would suggest
the false-negative problem wasn't as limiting in practice as the design
discussion predicted.

### Phase 1c — Scaffolding / connection-reuse cost

**What it adds:** connection cost that depends not just on distance (or hop
count, post-1b) but on whether a path already has established traffic along
it — new connections preferentially form along existing corridors rather than
through unscaffolded regions, mirroring how pioneer axons and radial glia
provide reusable physical substrate for later connections.

**Why this is third:** it's an independent refinement from relay decomposition
(one is about *how* distance is traversed, the other is about *whether reuse
makes it cheaper) and testing it against a 1a/1b baseline that already has
relay decomposition isolates its effect cleanly.

**Shared risk from 1a freeze:** preferential attachment along existing
corridors does not by itself guarantee that **new** edges/units on those
corridors establish under gradient pressure; same non-establishment risk as
1a/1b if the net is already near-ceiling on the objective. See
`docs/phase1a_results.md`.

**Success criterion:** connections cluster into reused, shared corridors
rather than being independently and uniformly distance-priced point-to-point.
This is a structural/qualitative prediction (a specific, checkable geometric
pattern), not just a performance number — worth having both.

### Phase 1d — Stretch: connection-cost-driven specialization (optional, not committed)

**What it might add, if 1a–1c succeed:** a direct test of whether functional
specialization emerges as a side effect of connection-cost pressure alone,
without ever rewarding specialization directly — the same design principle
behind the connection-cost-drives-modularity literature (below). This would
also be the natural point to test genuine expanding-capacity class addition
(flagged as untested in Phase 0's digit-split experiment, which only used a
fixed-size output head) — since real growth mechanics would finally exist to
test it against, rather than needing a separate isolated experiment.

Not committed as a specific implementation yet — revisit once 1a–1c have
results, since what's worth testing here depends on what those show.

---

## 4. Reasoning: why this shape, not another

**Why pressure over encoding, throughout:** the entire redirection of this
project — from PSMCL's self-model components to a developmental-milestone
framing to "encode the pressures, not the properties" — rests on one
empirical pattern found across the literature review: modularity, forgetting-
resistance, and morphological complexity all showed up as side effects of
evolutionary or training pressure (connection cost, resource scarcity) in the
studies reviewed, not as directly-selected-for targets. Phase 1's phasing is
built to keep testing that same pattern rather than re-introducing hand-coded
structure by another name.

**Why homeostatic, not hotspot-seeking, growth:** a controlled ablation
comparing growth-with-pruning against growth-only found the growth-only
variant reached similar task performance at roughly two orders of magnitude
more parameters — direct evidence that an unconstrained, unidirectional
growth rule doesn't converge, it balloons. Biological structural plasticity
models use a bidirectional, target-seeking rule (deviation from a target
firing rate drives expansion or shrinkage in either direction) specifically
to avoid this. Phase 1a's homeostatic requirement is a direct response to
that evidence, not a stylistic preference.

**Why relay/landmark decomposition, not a stronger single gradient:**
axon-guidance research is explicit that diffusible chemical gradients are
only informative over a bounded range, and that the actual solution to
long-range wiring is dividing the trajectory into short segments, each
governed by its own local target ("guidepost" cells), rather than one
gradient spanning the whole distance. This was checked directly against the
"just strengthen the broadcast signal" alternative: under multiplicative
per-hop decay, effective range grows only logarithmically with source
amplitude, while amplitude increases compress local dynamic range near the
source — a real, worsening trade-off in the opposite direction from what
relay decomposition achieves by keeping every hop local and short. Biology
converged on decomposition, not amplification, for a structural reason, not
an incidental one, and Phase 1 follows that reasoning rather than the
initial, simpler propagation-decay design.

**Why scaffolding reuse as its own, separate mechanism:** pioneer axons and
radial glia provide a physical substrate that later connections reuse rather
than re-deriving direction from scratch — a mechanism with no analog in the
original distance-cost design at all. It predicts a specific, different
outcome (clustering into reused corridors) than pure distance pricing does,
which is why it's tested as an independent addition rather than folded into
1b.

**Why structural decisions are governed by a local plasticity rule instead of
an optimizer over topology:** the broadcast-error learning rule validated in
Phase 0 (and its close relatives — self-referential weight matrices,
three-factor Hebbian rules) has a real mathematical guarantee behind it, but
that guarantee is specifically about continuous weight updates being
sign-matched to the true gradient. Discrete architectural decisions (grow
this, prune that) are a different mathematical object — a policy over
actions, not a continuous update — and nothing in that theory extends
automatically. This was the same problem HyperNCA hit: its actual training
regime was evolutionary search over whole task variants, not gradient
descent, precisely because discrete structural edits aren't differentiable.
Rather than solving that credit-assignment problem, Phase 1 sidesteps it:
real nervous systems don't appear to optimize topology via gradient descent
or RL either — they use local mechanisms (Hebbian growth, synaptic
competition, homeostatic plasticity, activity-dependent spine formation and
pruning) where each unit's structural fate is decided from its own local
history, with no signal ever flowing back from a global objective. Phase 0's
own result is direct empirical support for expecting this to work
reasonably well despite the missing guarantee: `activation`, a signal with
zero awareness of task loss, was nonetheless a robust proxy for actual
gradient-based importance across every condition tested. The trade is
explicit: no guarantee that structural change optimizes the same objective
as the weights, in exchange for a mechanism that's local by construction and
has a real, if indirect, empirical reason to land somewhere sensible.

**Why the plasticity rule's own set-point can adapt without this becoming
task-level optimization — the metaplasticity distinction:** biological
plasticity is not perfectly fixed across a lifetime, but what changes it is
still local and non-task-aware. Metaplasticity — literally "the plasticity
of synaptic plasticity" — describes mechanisms like the BCM sliding
threshold, where a neuron's own recent average firing rate sets the boundary
between strengthening and weakening future synapses, with the specific,
established function of stabilizing otherwise-unstable Hebbian plasticity.
This resolves what would otherwise be a false binary (fixed rule vs.
task-optimized rule) into three genuinely distinct tiers, each operating on
a different timescale and each worth naming explicitly since Phase 1 uses
more than one of them:

1. **Evolutionary tier — across experiments.** Manual iteration on the rule's
   design (this project's own cadence: propose a rule variant, run it,
   inspect results, adjust) is the right analogy for how biological plasticity
   rules themselves were shaped by natural selection — slow, informed by
   outcomes after the fact, not differentiable or continuous.
2. **Metaplastic tier — within a single run.** A sliding, locally-adaptive
   threshold (tested as a design variant in Phase 1a) operates faster than
   the evolutionary tier but is still local and still has no task-level
   objective — it adapts the rule's own operating point based on a cell's
   own history, mirroring BCM's demonstrated stabilizing role.
3. **Synaptic tier — moment to moment.** The Hebbian/homeostatic growth,
   competition, and pruning rules themselves, already specified in Phase 1a.

Phase 1a is scoped to directly compare a design with only tiers 1 and 3
(fixed threshold) against a design with all three tiers (sliding threshold),
specifically on the stability diagnostic that a missing credit-assignment
guarantee makes more important than it would otherwise be. **Diagnostic
freeze caveat:** that A/B only answers the metaplasticity question if growth
can establish under either tier set; pre-primary results show it does not —
see Phase 1a status box and `docs/phase1a_results.md`.

---

## 5. Supporting references

Grouped by what each is cited for. Where a precise URL was confirmed via
search during this project, it's included; where a citation is a named
concept discussed and evaluated but the specific source link wasn't retained
verbatim, it's marked for re-verification before formal write-up.

**Homeostatic structural plasticity / connection-cost-driven modularity**
(basis for Phase 1a's homeostatic requirement and 1d's specialization
hypothesis): computational and neuroscience literature on activity-dependent
structural plasticity (firing-rate deviation from a target driving neurite
field growth/shrinkage) and on connection-cost minimization producing
modularity as a side effect of a wiring-cost objective, without directly
selecting for specialization — discussed and evaluated in depth earlier in
this project; re-verify specific paper citations before external write-up.

**Metaplasticity / sliding threshold** (basis for Phase 1a's fixed-vs-adaptive
threshold design variant): the BCM (Bienenstock-Cooper-Munro) model and its
sliding modification threshold, and metaplasticity as the general phenomenon
of activity-dependent change in plasticity rules themselves:
- Abraham, "Metaplasticity: tuning synapses and networks for plasticity,"
  Nature Reviews Neuroscience — https://www.nature.com/articles/nrn2356
- Scholarpedia, "Metaplasticity" (BCM sliding threshold mechanism and
  NMDA-receptor-subunit basis) — http://www.scholarpedia.org/article/Metaplasticity
- "Mechanisms of heterosynaptic metaplasticity," PMC —
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3843880/
- "Calcium-Dependent But Action Potential-Independent BCM-Like Metaplasticity
  in the Hippocampus," Journal of Neuroscience —
  https://www.jneurosci.org/content/32/20/6785
- "Theories of synaptic memory consolidation and intelligent plasticity for
  continual learning" (metaplasticity's stabilizing role for Hebbian
  plasticity specifically, directly relevant to the stability diagnostic
  above) — https://arxiv.org/pdf/2405.16922
- "Computational models of learning and synaptic plasticity" (BCM formalism,
  §5.3 Metaplasticity) — https://arxiv.org/pdf/2412.05501

**Axon guidance / long-range wiring from local rules** (basis for Phase 1b/1c):
- eLife, "Axon Guidance: Watching axons on the move" —
  https://elifesciences.org/articles/106190
- Current Biology, axon guidance overview (gradient vs. short-range strategies) —
  https://www.cell.com/current-biology/fulltext/S0960-9822(99)80492-8
- Kaprielian et al., "Axon guidance at the midline choice point," Developmental
  Dynamics (2001) — https://anatomypubs.onlinelibrary.wiley.com/doi/10.1002/dvdy.1143
- ScienceDirect Topics, "Axon Guidance" overview (guidepost cells, short- vs.
  long-range cues) — https://www.sciencedirect.com/topics/immunology-and-microbiology/axon-guidance
- Tessier-Lavigne & Goodman, "The Molecular Biology of Axon Guidance," Science —
  https://people.physics.illinois.edu/Selvin/PRS/498IBR/The%20Molecular%20Biology%20of%20Axon%20Guidance%20--%20Tessier-Lavigne%20and%20Goodman%20274%20%285290%29_%201123%20--%20Science.pdf

**NCA-as-substrate precedent** (basis for treating NCA as a viable orchestrator
mechanism at all): Growing Neural Cellular Automata (Distill.pub, Mordvintsev
et al.) as the foundational example of local-rule-driven morphogenesis;
HyperNCA (NCA hypernetworks growing RL policy networks, including a
metamorphosis variant that transforms weights across task variants) as the
closest existing precedent for "one network's structure grown/modified by an
NCA," and as the source of the discrete-credit-assignment caveat above (its
actual training regime was evolutionary, not gradient-based) — re-verify
specific paper links before external write-up.

**Local credit assignment without full gradient access** (basis for Phase 0's
validated signal and for Phase 1's constraint that the orchestrator never
sees gradients): self-referential weight matrices (Schmidhuber) and
Backpropamine (neuromodulated plasticity, trainable end-to-end) as the
general precedent for a component that edits another component's parameters
using only a broadcast signal; the specific "Credit Assignment Through
Broadcasting a Global Error Vector" (GEVB) result, with its proven sign-
matching guarantee for continuous updates — validated as a real, working
mechanism in Phase 0, and the direct source of the discrete-vs-continuous
caveat in Section 2 above. Re-verify specific paper links before external
write-up.

**Developmental framing / pressure-over-encoding philosophy**: the
developmental-milestone reframe and "encode the pressures, not the
properties" principle that motivated this entire project's redirection away
from PSMCL's original component-based self-model design — internal project
history, not an external citation.

---

## 6. What this document does not decide

- Exact hyperparameters, thresholds, or pass/fail numbers for any Phase 1
  stage — those belong in each stage's own implementation spec, written
  immediately before that stage begins, following the same pre-registration
  discipline as Phase 0.
- Whether Phase 1d happens at all, or what it looks like if it does — deferred
  until 1a–1c produce results to build on.
- Any change to the core design commitments in Section 2 — those are treated
  as fixed for the whole of Phase 1 unless a specific, deliberate decision is
  made to revisit them, which should be flagged as an architecture change,
  not folded quietly into a later stage's spec.
