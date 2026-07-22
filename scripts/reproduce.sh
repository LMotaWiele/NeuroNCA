#!/usr/bin/env bash
# Minimal smoke reproduce for CI / install check.
# Full chapter re-runs: see README.md (project closed — use new results roots).
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -d .venv ]]; then
  python -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
if [[ "${SMOKE:-1}" == "1" ]]; then
  python scripts/run_phase0.py --config configs/default.yaml --smoke --condition stationary
  python scripts/run_phase1a.py --config configs/phase1a_fixed.yaml --smoke --condition stationary \
    --results-root results/phase1a_smoke_repro
  echo "Smoke complete (Phase 0 + Phase 1a). Project is closed; see README.md."
else
  python scripts/run_phase0.py --config configs/default.yaml
  echo "Phase 0 full complete. See results/phase0/aggregate/verdict.json"
fi
