# PGFS++

**Molecular Improvement Ensuring Synthetic Accessibility under Diversity Constraints**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB.svg)](https://www.python.org/)
[![PyTorch 2.3](https://img.shields.io/badge/PyTorch-2.3%20%2B%20CUDA%2012.1-EE4C2C.svg)](https://pytorch.org/)

Official implementation of **PGFS++**, a hierarchical PPO policy that improves a starting molecule by applying reaction templates and optional second reactants. The action space is a library of chemically valid reactions, so every trajectory is a synthesis route. An input–output similarity bonus keeps the product close to the input while still improving the target property (QED or sEH).

This repository is self-contained: reactant pools, reaction templates, the T→R2 mask cache, the sEH proxy, and compact evaluation logs are included.

<p align="center">
  <img src="assets/qed_panel.png" width="280" alt="PGFS++ compact ΔQED evaluation panel">
</p>

<p align="center">
  <em>Shipped compact eval on the 2k test set (ΔQED, diversity, input–output similarity).</em>
</p>

---

## What is included

| Path | Contents |
|------|----------|
| `src/` | Hierarchical PPO trainer, reaction environment, rewards |
| `configs/` | Paper configs (`delta_qed_*`, `delta_seh_*`) and a smoke config |
| `data/` | Train pool, 2k test set, templates, precomputed T→R2 masks |
| `scoring/seh/` | Bengio et al. (2021) sEH proxy weights |
| `run_detailed_results/compact/` | Greedy trajectories used by the default plots |
| `eval/`, `visualization/` | Compact eval and panel-plot launchers |

Paper checkpoints (~91 MB each) are optional and downloaded separately. Default plots do **not** need them.

---

## Installation

Tested on Ubuntu with an NVIDIA GPU, CUDA 12.1, PyTorch 2.3.0, and RDKit 2023.9.6 from conda-forge.

```bash
git clone https://github.com/bz317/PGFS_plus_plus.git
cd PGFS_plus_plus

conda env create -f conda_env_pgfspp.yml
conda activate pgfspp
bash scripts/bootstrap_pgfspp_cuda121.sh
pip install -e . --no-deps
```

`--no-deps` keeps the conda / CUDA wheels already installed by the bootstrap script. Use `SKIP_PYG=1 bash scripts/bootstrap_pgfspp_cuda121.sh` for a ΔQED-only setup; ΔSEH needs PyTorch Geometric.

**Check the install** (loads shipped data, parses a molecule, reports CUDA):

```bash
conda activate pgfspp
python scripts/check_install.py
```

You should see `PGFS++ install check passed.`

---

## Quick start: smoke training

Run one short ΔQED PPO update on the real data files. This is the fastest way to confirm that training can start; it is **not** a paper result.

```bash
conda activate pgfspp
bash scripts/run_smoke.sh
```

The smoke config (`configs/smoke_delta_qed.yaml`) uses 64 environment steps, one PPO epoch, and an 8-molecule greedy eval. On a single A100 it finishes in well under a minute and writes `runs/<id>/final_model.pt`. RDKit kekulize warnings during rollouts are expected and can be ignored.

Weights & Biases is optional. Training defaults to offline (`WANDB_MODE=disabled`) unless you set `WANDB_API_KEY`.

---

## Training

All commands below are run from the repository root with `conda activate pgfspp`.

Paper configs use a 4-step reaction budget, 1M environment steps, and an input–output similarity bonus (multiplicative in `*_scale.yaml`, additive in `*_weights.yaml`).

```bash
bash run_train.sh configs/delta_qed_scale.yaml
bash run_train.sh configs/delta_seh_scale.yaml
```

The other paper configs are `configs/delta_qed_weights.yaml` and `configs/delta_seh_weights.yaml`.

Equivalent entrypoint:

```bash
PYTHONPATH=. python -m src.scripts.run_experiment --config configs/delta_qed_scale.yaml
```

Checkpoints are written to `runs/<run-id>/`. To log to W&B:

```bash
export WANDB_MODE=online
export WANDB_API_KEY=...
export WANDB_ENTITY=...
bash run_train.sh configs/delta_qed_scale.yaml
```

The T→R2 mask cache is already in `data/r2_valid_indices.npz`. Rebuild it only if you change the reactant pool or templates:

```bash
bash preprocessing/run_precompute_r2_valid_indices.sh
```

---

## Evaluation

Shipped greedy trajectories for the fixed 2k test set (`data/reactants_test.pkl`) live under `run_detailed_results/compact/`. **Default plots use these files** and do not require GPU or checkpoints.

```bash
bash eval/run_plot_delta_qed_scale.sh
bash eval/run_plot_delta_seh_scale.sh
```

Outputs: `run_detailed_results/plots/compact_qed_panel.{pdf,png}` and `compact_seh_panel.{pdf,png}`. The ΔQED panel is quick. The ΔSEH panel scores the 2k trajectories with the sEH proxy (needs the PyG stack) and takes on the order of one to two minutes.

### Optional: paper checkpoints

| Run ID | Reward | Checkpoint |
|--------|--------|------------|
| `ymhrz9yg` | ΔQED scale | `runs/ymhrz9yg/model_step_1001472.pt` |
| `9gj82ve1` | ΔSEH scale | `runs/9gj82ve1/model_step_1001472.pt` |

```bash
bash scripts/download_checkpoints.sh
bash eval/run_eval_delta_qed_scale.sh
bash eval/run_eval_delta_seh_scale.sh
```

Eval + plot together (needs the checkpoints for the eval step):

```bash
bash eval/run_all_compact.sh
```

---

## Citation

```bibtex
@inproceedings{pgfspp2026,
  title     = {PGFS++: Molecular Improvement Ensuring Synthetic Accessibility under Diversity Constraints},
  author    = {Zhang, Boqiao},
  year      = {2026}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
The sEH proxy under `scoring/seh/` follows Bengio et al., *Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation* (NeurIPS 2021).
