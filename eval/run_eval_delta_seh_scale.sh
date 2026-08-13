#!/usr/bin/env bash
# Greedy eval on the 2k test set. Example: bash eval/run_eval_delta_seh_scale.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DEVICE="${DEVICE:-cuda}"

RUN_ID=ncs94oq8
CHECKPOINT="${CHECKPOINT:-${ROOT}/runs/${RUN_ID}/model_step_1001472.pt}"
STARTS="${STARTS:-${ROOT}/data/reactants_test.pkl}"
OUT_DIR="${OUT_DIR:-${ROOT}/results}"
OUT_PREFIX="${OUT_PREFIX:-4s_delta_seh_${RUN_ID}_1m_compact}"

python3 "${ROOT}/tools/eval_detailed_trajectories_bi_ppo.py" \
  --checkpoint "${CHECKPOINT}" \
  --run-id "${RUN_ID}" \
  --starts-file "${STARTS}" \
  --out-dir "${OUT_DIR}" \
  --out-prefix "${OUT_PREFIX}" \
  --device "${DEVICE}" \
  --inference-mode greedy
