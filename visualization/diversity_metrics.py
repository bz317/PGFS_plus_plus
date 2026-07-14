"""Structural diversity helpers for PGFS++ plot caches and panels."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

FP_RADIUS = 2
FP_SIZE = 1024
PAIRWISE_DENSITY_SMOOTH_SIGMA = 2.5


def _fp_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE).GetFingerprint(
        mol
    )


def internal_diversity(fps: list) -> float:
    """1 - mean pairwise Tanimoto similarity."""
    n = len(fps)
    if n < 2:
        return float("nan")
    sim_sum = 0.0
    n_pairs = 0
    for i in range(n - 1):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
        sim_sum += float(sum(sims))
        n_pairs += len(sims)
    if n_pairs == 0:
        return float("nan")
    return 1.0 - sim_sum / n_pairs


def pairwise_diversity_distances(fps: list) -> np.ndarray:
    """All ``N (N - 1) / 2`` pairwise Tanimoto distances (``1 - similarity``)."""
    n = len(fps)
    n_pairs = n * (n - 1) // 2
    if n_pairs == 0:
        return np.array([], dtype=np.float32)
    out = np.empty(n_pairs, dtype=np.float32)
    offset = 0
    for i in range(n - 1):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
        chunk = np.subtract(1.0, sims, dtype=np.float32)
        m = len(chunk)
        out[offset : offset + m] = chunk
        offset += m
    return out


def pairwise_distances_summary(distances: np.ndarray) -> dict[str, float]:
    if len(distances) == 0:
        return {
            "n_pairs": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "q25": float("nan"),
            "median": float("nan"),
            "q75": float("nan"),
        }
    return {
        "n_pairs": int(len(distances)),
        "mean": float(np.mean(distances)),
        "std": float(np.std(distances)),
        "min": float(np.min(distances)),
        "max": float(np.max(distances)),
        "q25": float(np.percentile(distances, 25)),
        "median": float(np.percentile(distances, 50)),
        "q75": float(np.percentile(distances, 75)),
    }


def _smooth_1d(values: np.ndarray, *, sigma: float) -> np.ndarray:
    if sigma <= 0 or len(values) < 3:
        return values.astype(float)
    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(values.astype(float), kernel, mode="same")


def plot_cell_violin_pairwise_distances(
    ax: plt.Axes,
    distances: Path | np.ndarray,
    *,
    color: str,
    y_min: float = 0.0,
    y_max: float = 1.0,
    n_bins: int = 64,
    summary: dict[str, float] | None = None,
    show_quartile_box: bool = True,
    half_width: float = 0.38,
    fill_alpha: float = 0.85,
    edgecolor: str = "#333333",
) -> None:
    """Violin from the full pairwise-distance array."""
    if isinstance(distances, Path):
        arr: np.ndarray = np.load(distances, mmap_mode="r")
    else:
        arr = np.asarray(distances, dtype=np.float32)

    if len(arr) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylim(y_min, y_max)
        return

    hist, edges = np.histogram(arr, bins=n_bins, range=(y_min, y_max), density=True)
    hist = _smooth_1d(hist, sigma=PAIRWISE_DENSITY_SMOOTH_SIGMA)
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak = float(hist.max()) if len(hist) else 0.0
    widths = (hist / peak * half_width) if peak > 0 else np.zeros_like(hist)
    ax.fill(
        np.concatenate([-widths, widths[::-1]]),
        np.concatenate([centers, centers[::-1]]),
        color=color,
        alpha=fill_alpha,
        edgecolor=edgecolor,
        linewidth=0.65,
    )

    if summary is None:
        summary = pairwise_distances_summary(arr)
    med = summary["median"]
    box_half = half_width * 0.10
    ax.plot([-box_half, box_half], [med, med], color="white", linewidth=2.0, zorder=5)
    if show_quartile_box:
        q1 = summary["q25"]
        q3 = summary["q75"]
        ax.plot([-box_half, box_half], [q1, q3], color="#333333", linewidth=1.2, zorder=5)
        ax.plot([-box_half, box_half], [q1, q1], color="#333333", linewidth=1.0, zorder=5)
        ax.plot([-box_half, box_half], [q3, q3], color="#333333", linewidth=1.0, zorder=5)

    ax.set_xlim(-half_width - 0.14, half_width + 0.14)
    ax.set_xticks([])
    ax.set_ylim(y_min, y_max)


def smiles_to_fps(smiles_list: list[str]) -> list:
    fps = []
    for smi in smiles_list:
        fp = _fp_from_smiles(smi)
        if fp is not None:
            fps.append(fp)
    return fps


def bootstrap_diversity(
    fps: list,
    *,
    sample_size: int,
    n_bootstrap: int,
    seed: int,
) -> list[float]:
    if len(fps) < 2:
        return []
    rng = np.random.default_rng(seed)
    n = len(fps)
    k = min(int(sample_size), n)
    values: list[float] = []
    for _ in range(int(n_bootstrap)):
        idx = rng.choice(n, size=k, replace=True)
        subset = [fps[int(i)] for i in idx]
        if len(subset) < 2:
            continue
        values.append(internal_diversity(subset))
    return values
