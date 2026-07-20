#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -d .venv ]]; then python -m venv .venv; fi
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/run_phase0.py --config configs/default.yaml --smoke
echo "Phase 0 complete. See results/phase0/aggregate/verdict.json"
