#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -d .venv ]]; then
  python -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -e ".[dev]"
# Primary full Phase 0 by default (seeds 0,1,2, full budget).
# Quick path: SMOKE=1 ./scripts/reproduce.sh
if [[ "${SMOKE:-0}" == "1" ]]; then
  python scripts/run_phase0.py --config configs/default.yaml --smoke
else
  python scripts/run_phase0.py --config configs/default.yaml
fi
echo "Phase 0 complete. See results/phase0/aggregate/verdict.json"
