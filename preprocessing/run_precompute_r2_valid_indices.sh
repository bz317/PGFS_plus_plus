#!/usr/bin/env bash
# Precompute T->R2 pattern masks (r2_available) for PGFS++.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS="${JOBS:-16}"
REACTANTS_PKL="${REACTANTS_PKL:-${ROOT}/data/reactants_train.pkl}"
TEMPLATES_PKL="${TEMPLATES_PKL:-${ROOT}/data/templates.pkl}"
OUTPUT_NPZ="${OUTPUT_NPZ:-${ROOT}/data/r2_valid_indices.npz}"
MANIFEST_JSON="${MANIFEST_JSON:-${ROOT}/data/r2_valid_indices_manifest.json}"

PY_ARGS=(
  --reactants-pkl "${REACTANTS_PKL}"
  --templates-pkl "${TEMPLATES_PKL}"
  --output "${OUTPUT_NPZ}"
  --manifest "${MANIFEST_JSON}"
  --jobs "${JOBS}"
)
if [[ "${SKIP_VERIFY:-0}" != "1" ]]; then
  PY_ARGS+=(--verify)
fi

python3 "${ROOT}/preprocessing/precompute_r2_valid_indices.py" "${PY_ARGS[@]}"
