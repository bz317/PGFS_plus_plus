# Checkpoints

Paper checkpoints are optional. Plots use the shipped detailed trajectories under
`run_detailed_results/compact/` and do not need these files.

| Run ID | Reward | Config | Checkpoint |
|--------|--------|--------|------------|
| `ymhrz9yg` | ΔQED scale | `configs/delta_qed_scale.yaml` | `model_step_1001472.pt` |
| `9gj82ve1` | ΔSEH scale | `configs/delta_seh_scale.yaml` | `model_step_1001472.pt` |

Download:

```bash
bash scripts/download_checkpoints.sh
```

Expected layout after download:

```
runs/ymhrz9yg/model_step_1001472.pt
runs/9gj82ve1/model_step_1001472.pt
```
