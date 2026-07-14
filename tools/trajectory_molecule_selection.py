"""Per-episode molecule selection rules for QED, SA, and diversity plots."""

from __future__ import annotations

from pathlib import Path

MOLECULE_SELECTION = "final"

# SA-score JSON tree (mirrors compute_final_molecule_sa.py outputs).
SELECTION_SA_ROOT: dict[str, str] = {
    "final": "run_detailed_results/sa_scores",
}


def selection_for_method(method: str, *, dataset: str | None = None) -> str:
    """Return the terminal molecule selection used for all methods."""
    del method, dataset
    return MOLECULE_SELECTION


def _iter_block_rows(path: Path, block: str):
    """Yield ``dict`` rows for a tab-separated ``[block]`` in a detailed log."""
    in_block = False
    header: list[str] | None = None
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line == f"[{block}]":
                in_block = True
                continue
            if not in_block:
                continue
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                break
            parts = line.split("\t")
            if header is None:
                header = parts
                continue
            if len(parts) < len(header):
                parts += [""] * (len(header) - len(parts))
            yield dict(zip(header, parts))


def parse_trajectory_start_episodes(path: Path) -> tuple[list[str], list[float]]:
    """Per-episode ``(start_smiles, initial_qed)`` from ``[trajectories]``."""
    smiles: list[str] = []
    qeds: list[float] = []
    for row in _iter_block_rows(path, "trajectories"):
        smi = str(row.get("start_smiles", "")).strip()
        qed_str = str(row.get("initial_qed", "")).strip()
        if not smi or not qed_str:
            continue
        try:
            qeds.append(float(qed_str))
        except ValueError:
            continue
        smiles.append(smi)
    return smiles, qeds


def parse_trajectory_final_smiles(path: Path) -> list[str]:
    """SMILES of the terminal molecule of each trajectory (``final_smiles``)."""
    smiles: list[str] = []
    for row in _iter_block_rows(path, "trajectories"):
        smi = str(row.get("final_smiles", "")).strip()
        if smi:
            smiles.append(smi)
    return smiles


def parse_trajectory_smiles(path: Path, selection: str = MOLECULE_SELECTION) -> list[str]:
    """SMILES per trajectory (terminal molecule only)."""
    if selection != MOLECULE_SELECTION:
        raise ValueError(
            f"selection must be {MOLECULE_SELECTION!r}, got {selection!r}"
        )
    return parse_trajectory_final_smiles(path)


def parse_trajectory_qed(path: Path, selection: str = MOLECULE_SELECTION) -> list[float]:
    """Per-episode final QED from the ``[trajectories]`` block."""
    if selection != MOLECULE_SELECTION:
        raise ValueError(
            f"selection must be {MOLECULE_SELECTION!r}, got {selection!r}"
        )
    values: list[float] = []
    for row in _iter_block_rows(path, "trajectories"):
        qed_str = str(row.get("final_qed", "")).strip()
        if not qed_str:
            continue
        try:
            values.append(float(qed_str))
        except ValueError:
            continue
    return values


def parse_trajectory_initial_qed(path: Path) -> list[float]:
    """Per-episode start QED from the ``[trajectories]`` block."""
    values: list[float] = []
    for row in _iter_block_rows(path, "trajectories"):
        qed_str = str(row.get("initial_qed", "")).strip()
        if not qed_str:
            continue
        try:
            values.append(float(qed_str))
        except ValueError:
            continue
    return values


def parse_trajectory_delta_qed(path: Path) -> list[float]:
    """Per-episode delta QED from the ``[trajectories]`` block."""
    values: list[float] = []
    for row in _iter_block_rows(path, "trajectories"):
        delta_str = str(row.get("delta_qed", "")).strip()
        if not delta_str:
            continue
        try:
            values.append(float(delta_str))
        except ValueError:
            continue
    return values


def parse_trajectory_method_qed(
    path: Path,
    method: str,
    *,
    dataset: str | None = None,
) -> tuple[list[float], str]:
    """Per-episode QED for ``method`` plus the selection that was applied."""
    del method, dataset
    return parse_trajectory_qed(path), MOLECULE_SELECTION


def parse_trajectory_smiles_for_method(
    path: Path,
    method: str,
    *,
    dataset: str | None = None,
) -> tuple[list[str], str]:
    """SMILES for ``method`` plus the selection that was applied."""
    del method, dataset
    return parse_trajectory_smiles(path), MOLECULE_SELECTION
