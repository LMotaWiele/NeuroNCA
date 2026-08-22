# Notes index — read this first

Agents: load **this file only**, then the **one** row matching the task. Do not bulk-read notes/.

| Topic | File | Status | One-liner |
|---|---|---|---|
| How to write notes | [README.md](README.md) | meta | layers, front-matter, access rules |
| Layout exceptions | [docs/design/layout.md](../../docs/design/layout.md) | sealed | configs/, data/, results/; nested bench worktrees are not product |
| Engineering invariants | [docs/design/invariants.md](../../docs/design/invariants.md) | sealed | GT isolation, RNG, ledger freeze, local-only plasticity, grid neighbors |

## Glossary (optional)

| Name | Means |
|---|---|
| `PARETO_BOTH_FAIL` | Stationary MNIST: dense already flat; plastic not competitive |
| `RATE_CAP_RESOLVES_THRASH` | Fashion thrash closed by `max_new_neurons_per_layer=1` |
| `ADAPT_COST_FIXED_WINS` | Mild shifts: everyone recovers in 1 ep; plastic costs wall-clock |
| `RETENTION_FIXED_WINS` | Split→restore: fixed retains more of A than plastic |
| `FEATURE_NEIGHBORS_WORSE` | Feature-space neighbors did not beat grid on stationary bars |
| `MIXED_LAYER_DEPENDENT` | Spatial specialization coherence is not a full regional-consolidation build |

## Active next

- In-scope chapters are **closed**. Reproduce from frozen trees / documented scripts; do not overwrite primaries.
- **Do not:** edit `neuronca-bench-t*` worktrees; reopen Phase 1b/1c or a harder-shift treadmill without a new product mandate.
