#!/usr/bin/env bash
# One short ΔQED PPO update to confirm training can start.
#   bash scripts/run_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_PROJECT="${WANDB_PROJECT:-PGFS++}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

python3 -m src.scripts.run_experiment --config "${ROOT}/configs/smoke_delta_qed.yaml"
