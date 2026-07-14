# PGFS++

Official implementation of **PGFS++** — Molecular Improvement Ensuring Synthetic Accessibility under
Diversity Constraints.

---

## Folder structure

After installation and data setup the workspace root should look like this:

```
PGFS++/
├── src/                        # PPO trainer, chemistry, rewards
├── configs/                    # Experiment YAMLs (ΔQED / ΔSEH)
├── data/                       # Train pool, 2k test set, templates, R2 cache
├── scoring/                    # Reward models (e.g. sEH proxy weights)
├── preprocessing/              # T-R2 masking pre-compute
├── tools/                      # Detailed trajectory eval utilities
├── visualization/            # Plot scripts
├── eval/                       # Eval and plot shell launchers
├── runs/                       # Training checkpoints
└── run_detailed_results/       # Detailed trajectories and plot outputs
```

| Folder | Purpose |
|--------|---------|
| **src/** | PGFS++ trainer (`ppo_bi`), chemistry helpers, and experiment entrypoint. |
| **configs/** | Four DINGOS-template configs: `delta_qed_*` and `delta_seh_*` (weights / scale). |
| **data/** | `reactants_train.pkl`, `reactants_test.pkl` (2k eval set), `templates.pkl`, `r2_valid_indices.npz`. |
| **scoring/** | `seh/bengio2021flow_proxy.pkl.gz` for the ΔSEH reward. |
| **preprocessing/** | Regenerate T→R2 masks if needed; shipped cache is under `data/`. |
| **eval/** | Greedy eval and plotting launchers for the compact 2k test set. |
| **runs/** | Checkpoints written during training; two 1M-step runs are shipped. |
| **run_detailed_results/** | Greedy detailed trajectories (`compact/`) and generated plots. |

---

## Installation

Our code is tested on Ubuntu with NVIDIA GPUs. We use CUDA 12.1, PyTorch 2.3.0, and RDKit from conda-forge.

```bash
cd PGFS++
conda env create -f conda_env_pgfspp.yml
conda activate pgfspp
bash scripts/bootstrap_pgfspp_cuda121.sh
pip install -e .
```

**Sanity check:**

```bash
conda activate pgfspp
PYTHONPATH=. python -m compileall -q src tools visualization
python -c "import torch, rdkit, wandb; print('ok', torch.__version__)"
```

**Update an existing environment:**

```bash
conda activate pgfspp
conda env update -n pgfspp -f conda_env_pgfspp.yml --prune
bash scripts/bootstrap_pgfspp_cuda121.sh
```

> **Note:** `delta_seh_*` configs require PyTorch Geometric (installed by the bootstrap script). `delta_qed_*` training and eval need only PyTorch and RDKit.

---

## Training

All commands are run from `PGFS++/`. Example configs use 4 reaction step budget, 1M training steps, and input-output similarity bonus (additive or multiplicative).

```bash
conda activate pgfspp
export WANDB_MODE=disabled   # or set WANDB_PROJECT / WANDB_ENTITY

bash run_train.sh configs/delta_qed_scale.yaml
bash run_train.sh configs/delta_seh_scale.yaml
```

Other configs: `configs/delta_qed_weights.yaml`, `configs/delta_seh_weights.yaml`.

Manual entrypoint:

```bash
PYTHONPATH=. python -m src.scripts.run_experiment --config configs/delta_qed_scale.yaml
```

Checkpoints are saved under `runs/<wandb_run_id>/`.

**Optional:** regenerate the T-R2 mask cache (shipped as `data/r2_valid_indices.npz`):

```bash
bash preprocessing/run_precompute_r2_valid_indices.sh
```

---

## Evaluation

Evaluate shipped checkpoints with greedy rollouts on the fixed 2k test set (`data/reactants_test.pkl`), then build violin / metrics panels.

| Run ID | Reward | Checkpoint |
|--------|--------|------------|
| `ymhrz9yg` | ΔQED scale | `runs/ymhrz9yg/model_step_1001472.pt` |
| `9gj82ve1` | ΔSEH scale | `runs/9gj82ve1/model_step_1001472.pt` |

Reference detailed results are under `run_detailed_results/compact/`.

**Greedy eval** (re-run trajectories; needs GPU):

```bash
bash eval/run_eval_delta_qed_scale.sh
bash eval/run_eval_delta_seh_scale.sh
```

**Plots** (default: shipped detailed results, Ours only):

```bash
bash eval/run_plot_delta_qed_scale.sh
bash eval/run_plot_delta_seh_scale.sh

# Or plot directly:
python visualization/plot_ours_compact_panel.py --reward delta_qed
python visualization/plot_ours_compact_panel.py --reward delta_seh
```

**Eval + plot together:**

```bash
bash eval/run_all_compact.sh
```

Outputs: `run_detailed_results/plots/compact_qed_panel.pdf`, `compact_seh_panel.pdf`.
