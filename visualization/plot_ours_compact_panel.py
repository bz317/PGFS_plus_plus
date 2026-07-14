#!/usr/bin/env python3
"""Plot PGFS++ (Ours) metrics from shipped detailed trajectory results.

Example: python visualization/plot_ours_compact_panel.py --reward delta_qed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.inout_sim import compute_inout_sim_from_detailed
from tools.pgfspp_plot_cache import build_ours_cache
from visualization.diversity_metrics import plot_cell_violin_pairwise_distances
from visualization.panel_style import (
    MLMI_HEADER_BLUE,
    MLMI_TEXT_DARK,
    MLMI_VIOLIN_ALPHA,
    MLMI_VIOLIN_EDGE,
    _apply_mlmi_rcparams,
    _plot_poster_cell_violin,
    _style_poster_axes,
)

METHOD = "Ours"
DATASET = "compact"

DEFAULT_DETAILED: dict[str, Path] = {
    "delta_qed": REPO
    / "run_detailed_results/compact/4s_delta_qed_ymhrz9yg_1m_compact_results.txt",
    "delta_seh": REPO
    / "run_detailed_results/compact/4s_delta_seh_9gj82ve1_compact_results.txt",
}
DEFAULT_OUTPUT: dict[str, Path] = {
    "delta_qed": REPO / "run_detailed_results/plots/compact_qed_panel",
    "delta_seh": REPO / "run_detailed_results/plots/compact_seh_panel",
}


def _metric_rows(reward: str) -> list[tuple[str, str, float, float, list[float] | None]]:
    obj_key = "delta_qed" if reward == "delta_qed" else "delta_seh"
    obj_label = r"$\Delta$QED ↑" if reward == "delta_qed" else r"$\Delta$SEH ↑"
    return [
        (obj_key, obj_label, -0.2, 1.0, None),
        ("diversity", "Diversity ↑", 0.0, 1.0, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        ("in_out_sim", "In/out sim ↑", 0.0, 1.0, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    ]


def _values(payload: dict, key: str, reward: str) -> list[float]:
    if key == "delta_qed":
        return [float(x) for x in (payload.get("delta_qed") or [])]
    if key == "delta_seh":
        return [float(x) for x in (payload.get("delta_seh") or [])]
    if key == "in_out_sim":
        return [float(x) for x in (payload.get("in_out_sim") or [])]
    raise KeyError(key)


def build_payload(
    *,
    detailed_path: Path,
    detailed_rel: str,
    reward: str,
    rebuild: bool,
) -> dict:
    cache_dir = REPO / "run_detailed_results/plot_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{reward}_ours.json"
    if cache_path.is_file() and not rebuild:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    payload = build_ours_cache(
        detailed_path=detailed_path,
        method=METHOD,
        dataset=DATASET,
        sample_size=2000,
        n_bootstrap=200,
        seed_offset=0,
    )
    payload["source_detailed_results"] = detailed_rel
    if reward == "delta_seh":
        from src.chem.seh_scorer import SehScorer
        from tools.trajectory_molecule_selection import _iter_block_rows

        scorer = SehScorer.from_config({"weights_path": "scoring/seh/bengio2021flow_proxy.pkl.gz"})
        deltas = []
        for row in _iter_block_rows(detailed_path, block="trajectories"):
            s = str(row.get("start_smiles", "")).strip()
            f = str(row.get("final_smiles", "")).strip()
            if s and f:
                deltas.append(float(scorer.reward(f)) - float(scorer.reward(s)))
        payload["delta_seh"] = deltas
        payload["median_delta_seh"] = float(np.median(deltas)) if deltas else None
    inout = compute_inout_sim_from_detailed(detailed_path)
    payload["in_out_sim"] = inout
    cache_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def plot_panel(*, payload: dict, reward: str, output_prefix: Path) -> None:
    _apply_mlmi_rcparams()
    rows = _metric_rows(reward)
    fig, axes = plt.subplots(len(rows), 1, figsize=(3.2, 2.2 * len(rows)), sharex=True)
    if len(rows) == 1:
        axes = [axes]

    pw_path = REPO / "run_detailed_results/plot_cache" / payload.get(
        "pairwise_distances_file", ""
    )

    for ax, (key, label, ymin, ymax, yticks) in zip(axes, rows):
        _style_poster_axes(ax)
        ax.set_ylabel(label, color=MLMI_TEXT_DARK, fontsize=9)
        ax.set_ylim(ymin, ymax)
        if yticks:
            ax.set_yticks(yticks)

        if key == "diversity" and pw_path.is_file():
            plot_cell_violin_pairwise_distances(
                ax,
                np.load(pw_path),
                color=MLMI_HEADER_BLUE,
                y_min=ymin,
                y_max=ymax,
                fill_alpha=MLMI_VIOLIN_ALPHA,
                edgecolor=MLMI_VIOLIN_EDGE,
            )
        else:
            vals = _values(payload, key, reward)
            _plot_poster_cell_violin(
                ax,
                vals,
                color=MLMI_HEADER_BLUE,
                y_min=ymin,
                y_max=ymax,
            )
        ax.set_xticks([0.5])
        ax.set_xticklabels([METHOD])
        ax.set_xlim(0.0, 1.0)

    fig.suptitle(f"PGFS++ eval ({reward})", fontsize=10, color=MLMI_TEXT_DARK)
    fig.tight_layout()
    out_pdf = output_prefix.with_suffix(".pdf")
    out_png = output_prefix.with_suffix(".png")
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary_rows = []
    for key, label, *_ in rows:
        if key == "diversity":
            pw = payload.get("pairwise_distances_summary") or {}
            med = pw.get("median")
            if med is None or (isinstance(med, float) and np.isnan(med)):
                continue
            summary_rows.append({"metric": key, "median": float(med), "n": int(pw.get("n_pairs", 0))})
            continue
        vals = _values(payload, key, reward)
        if not vals:
            continue
        summary_rows.append(
            {"metric": key, "median": float(np.median(vals)), "n": len(vals)}
        )
    pd.DataFrame(summary_rows).to_csv(
        output_prefix.with_name(output_prefix.name + "_summary.csv"), index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reward",
        choices=["delta_qed", "delta_seh"],
        required=True,
        help="Reward type; selects default detailed-results and output paths",
    )
    parser.add_argument(
        "--detailed-results",
        type=Path,
        default=None,
        help="Detailed trajectory log (default: shipped compact result for --reward)",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output path stem without extension (default: run_detailed_results/plots/compact_*_panel)",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    detailed = (args.detailed_results or DEFAULT_DETAILED[args.reward]).resolve()
    output_prefix = (args.output_prefix or DEFAULT_OUTPUT[args.reward]).resolve()
    if not detailed.is_file():
        raise FileNotFoundError(detailed)
    try:
        detailed_rel = str(detailed.relative_to(REPO))
    except ValueError:
        detailed_rel = str(detailed)

    payload = build_payload(
        detailed_path=detailed,
        detailed_rel=detailed_rel,
        reward=args.reward,
        rebuild=args.rebuild_cache,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    plot_panel(payload=payload, reward=args.reward, output_prefix=output_prefix)
    print(f"wrote {output_prefix.with_suffix('.pdf')}", flush=True)


if __name__ == "__main__":
    main()
