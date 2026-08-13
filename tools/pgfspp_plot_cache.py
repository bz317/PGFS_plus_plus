"""Minimal plot cache builder for PGFS++ detailed results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tools.trajectory_molecule_selection import (
    parse_trajectory_delta_qed,
    parse_trajectory_initial_qed,
    parse_trajectory_smiles_for_method,
)
from visualization.diversity_metrics import (
    bootstrap_diversity,
    pairwise_distances_summary,
    pairwise_diversity_distances,
    smiles_to_fps,
)

CACHE_VERSION = 1
REPO = Path(__file__).resolve().parents[1]


def build_ours_cache(
    *,
    detailed_path: Path,
    method: str,
    dataset: str = "compact",
    sample_size: int = 2000,
    n_bootstrap: int = 200,
    seed_offset: int = 0,
) -> dict:
    detailed_path = detailed_path.resolve()
    smiles, _ = parse_trajectory_smiles_for_method(detailed_path, method, dataset=dataset)
    fps = smiles_to_fps(smiles)
    pairwise = pairwise_diversity_distances(fps)
    pw_file = f"{dataset}_ours_pairwise_distances.npy"
    pw_dir = REPO / "results/plot_cache"
    pw_dir.mkdir(parents=True, exist_ok=True)
    np.save(pw_dir / pw_file, pairwise)

    return {
        "version": CACHE_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "method": method,
        "molecule_selection": "final",
        "qed": [float(x) for x in parse_trajectory_method_qed_values(detailed_path)],
        "delta_qed": [float(x) for x in parse_trajectory_delta_qed(detailed_path)],
        "mean_delta_qed": float(np.mean(parse_trajectory_delta_qed(detailed_path) or [0])),
        "diversity_bootstrap": bootstrap_diversity(
            fps, sample_size=sample_size, n_bootstrap=n_bootstrap, seed=seed_offset
        ),
        "diversity_full_set": float(pairwise.mean()) if len(pairwise) else float("nan"),
        "pairwise_distances_file": pw_file,
        "pairwise_distances_summary": pairwise_distances_summary(pairwise),
        "pairwise_distances_n_pairs": int(len(pairwise)),
        "start_qed_median": float(np.median(parse_trajectory_initial_qed(detailed_path) or [0])),
    }


def parse_trajectory_method_qed_values(path: Path) -> list[float]:
    from tools.trajectory_molecule_selection import _iter_block_rows

    out: list[float] = []
    for row in _iter_block_rows(path, "trajectories"):
        try:
            out.append(float(row.get("final_qed", "nan")))
        except ValueError:
            continue
    return out
