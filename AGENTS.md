# Project rules

Agent-agnostic layout and conventions. Stacks on `~/.grok/AGENTS.md` (graph + explore spawn policy). Prefer **AGENTS.md** — no `CLAUDE.md` / `.claude/` for Grok.

## Overview

- **Purpose**: Closed research codebase for structural plasticity on small MLPs. The reasoning net trains with ordinary backprop; local activation signals observe training and, in Phase 1a, drive homeostatic grow/prune of hidden neurons and adjacent edges. In-scope chapters (Phase 0, Phase 1a, Fashion thrash, self-cal r*, adaptation, spatial/feature-neighbor checks) are finished.
- **Stack**: Python >= 3.12 (`pyproject.toml`, hatchling, `.venv`)
- **Package**: `neuronca` (`src/neuronca/`)
- **Build / test**: `source .venv/bin/activate` · `pip install -e '.[dev]'` · `pytest`

## Layout

```
NeuroNCA/
├── AGENTS.md              # this file
├── README.md              # paper companion: findings + how to reproduce
├── pyproject.toml         # Python project (>=3.12, hatchling)
├── .venv/                 # local venv (gitignored)
├── configs/               # experiment YAMLs (allowed exception — see layout.md)
├── src/neuronca/          # production code only
├── tests/                 # flat test_<topic>.py (see mapping below)
├── scripts/               # campaign runners + analyzers — never imported by src/
├── notebooks/             # demos/experiments — not production logic
├── docs/design/           # ADRs / design ("why") + layout.md + invariants.md
├── docs/                  # preregs, results write-ups
├── data/                  # dataset downloads (eval data root)
├── results/               # experiment trees (frozen paths; do not rename)
├── .agent/skills/         # shared SKILL.md (not under .grok/ only)
├── .agent/notes/          # durable findings — INDEX.md first (see rule 7)
├── .grok/agents/          # harness shadows (explore → deepseek-flash)
├── .grok/config.toml      # optional project overrides only
├── opencode.json          # OpenCode harness wiring (keep; not product)
└── .mcp.json              # optional extra MCP only (not CRG)
```

Rationale for buckets: `docs/design/layout.md` (read on demand). Engineering contracts: `docs/design/invariants.md`.

**Not product layout:** nested `neuronca-bench-t2/`, `neuronca-bench-t3/`, `neuronca-bench-t34/`, `neuronca-bench-t4/` are detached-HEAD git worktrees (frozen bench pins). Sibling copies live under `~/Documents/neuronca-bench-t*`. Do not edit, move, reformat, comment-strip, or `git worktree remove` them.

## Hard rules

1. New modules under `src/neuronca/…` only — never free-floating at repo root.
2. Tests stay **flat**: `tests/test_<topic>.py` matching neighbors (this repo does not mirror `src/` path-for-path). One style; mechanical.
3. `scripts/` = agent-runnable campaign runners / analyzers; `notebooks/` = demos. Neither imported by `src/`. Promote shared code into `src/`.
4. Design/ADRs → `docs/design/`. README stays the paper companion (findings + reproduce), not a ticket tracker.
5. Before a **new top-level directory**, stop and ask. Prefer the tree above. Do not rename `results/` (scripts and docs hardcode paths). Do not relocate frozen result trees.
6. `.agent/` = shared knowledge/skills. `.grok/` / `.opencode/` / `opencode.json` / `.mcp.json` = harness wiring — do not merge.
7. **Notes (cheap path):** before re-deriving a diagnosis, opening a design debate, or continuing prior investigation → read `.agent/notes/INDEX.md` only, then the **one** linked file. After a finding → update that note’s front-matter (+ ledger row if used). Do not bulk-read notes/ or dump chat. Shape: `.agent/notes/README.md` (on demand). Prefer `docs/design/` for sealed/spec authority over notes.
8. Small focused edits; match neighbors; no unsolicited docs; verify against source not README alone.
9. No commit/push/PR unless asked. No production logic in scripts/notebooks/notes. No full Phase 0/1a GPU campaigns unless asked.
10. CRG MCP is **global** (`~/.grok/config.toml`) — no project `.mcp.json` for the graph. Build once: `code-review-graph build --repo .`
11. **Comments** — 1–2 lines that say what it is and why. No preregistration essays, no chapter-history, no "why we didn't do X". Why-at-length lives in `docs/design/`. A constraint an editor would otherwise break may stay, still 1–2 lines.
12. **Worktrees:** never touch `neuronca-bench-t*` paths. They are frozen pins, not this product tree.
13. Changing signal, GT, or product-stack code after a freeze → **new experiment id** and a **new results root**. Do not overwrite frozen trees in place.

## New module

Create `src/neuronca/…` + a flat `tests/test_<topic>.py`. Architecture change → `docs/design/`. Durable finding → `.agent/notes/` (via INDEX + front-matter).
