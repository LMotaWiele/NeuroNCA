# Why this layout

Short rationale for the tree in `AGENTS.md`. Agents should follow AGENTS.md hard rules; this file is background only.

| Path | Why |
|------|-----|
| `src/neuronca/` | Single home for production code; scaffolding stays mechanical. |
| `tests/` (flat) | Existing style is `tests/test_<topic>.py`, not a `src/` path mirror. Keep that mapping so new tests follow neighbors without a bulk rename. |
| `scripts/` vs `notebooks/` | Scripts are re-runnable campaign runners and analyzers; notebooks are demos/experiments. Splitting avoids a catch-all dump. Neither is imported by `src/`. |
| `docs/design/` | ADRs, design docs, and invariants — plan/agents look here for prior decisions before proposing new architecture. |
| `docs/` | Preregistrations and results write-ups (paper companion material). |
| `configs/` | **Allowed exception.** Experiment YAMLs (Phase 0 / Phase 1a product stacks) are first-class inputs to the runners. They stay top-level rather than under `src/` so campaigns can be launched without treating YAML as package data. |
| `data/` | Eval data root (MNIST / Fashion-MNIST / KMNIST downloads). Equivalent to a bench data tree; do not merge into `src/neuronca/data/` (that package is loaders/tasks, not downloads). |
| `results/` | Eval/run root (equivalent to `bench/` / `runs/`). **Do not rename** — scripts, diagnostics indexes, and write-ups hardcode these paths. Frozen trees stay in place; new campaigns need a new results root. Gitignore keeps bulk run artifacts out while tracking `results/.gitkeep` and the diagnostics README. |
| `.agent/notes/` | Hard-won findings; **INDEX.md** is the agent entrypoint, then one linked note. Shape in notes README. |
| `.agent/skills/` | Shared skills once, not cloned under every harness. |
| `.grok/` vs `.agent/` | Model routing and permissions differ per harness; knowledge should not. |
| `opencode.json` | OpenCode harness wiring used in this checkout. Keep; do not treat as product layout. |
| No project CRG `.mcp.json` | Graph server is global; project MCP only for extra servers. |

## Nested bench worktrees (not product)

`neuronca-bench-t2/`, `neuronca-bench-t3/`, `neuronca-bench-t34/`, and `neuronca-bench-t4/` are nested git worktrees (gitdir files pointing at `.git/worktrees/…`). Sibling checkouts also exist under `~/Documents/neuronca-bench-t*`.

They are **frozen detached-HEAD bench pins**, not part of the product tree. Leave them completely untouched: no edits, moves, reformats, comment sweeps, or `git worktree remove`. If they appear in a directory listing, ignore them.
