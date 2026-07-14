#!/usr/bin/env bash
# Re-run compact eval + plot for both AAAI "Ours" checkpoints.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${DIR}/run_eval_delta_qed_scale.sh"
bash "${DIR}/run_eval_delta_seh_scale.sh"
bash "${DIR}/run_plot_delta_qed_scale.sh" --rebuild-cache
bash "${DIR}/run_plot_delta_seh_scale.sh" --rebuild-cache
