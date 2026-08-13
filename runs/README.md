# Checkpoints

Paper checkpoints are used by the greedy eval scripts. Download:

```bash
bash scripts/download_checkpoints.sh
```

| Run ID | Reward | Config | Checkpoint |
|--------|--------|--------|------------|
| `ymhrz9yg` | ΔQED scale | `configs/delta_qed_scale.yaml` | `model_step_1001472.pt` |
| `ncs94oq8` | ΔSEH scale | `configs/delta_seh_scale.yaml` | `model_step_1001472.pt` |

Expected layout after download:

```
runs/ymhrz9yg/model_step_1001472.pt
runs/ncs94oq8/model_step_1001472.pt
```
