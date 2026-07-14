#!/usr/bin/env bash
# Train PGFS++ (PPO hierarchical lookup). Example:
#   bash run_train.sh configs/delta_qed_weights.yaml
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:?usage: run_train.sh <config.yaml>}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_PROJECT="${WANDB_PROJECT:-PGFS++}"

python3 -m src.scripts.run_experiment --config "${CONFIG}"
