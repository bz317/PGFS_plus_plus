"""Per-episode input–output Tanimoto similarity from detailed trajectory logs."""

from __future__ import annotations

from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from tools.trajectory_molecule_selection import _iter_block_rows

RDLogger.DisableLog("rdApp.*")

FP_RADIUS = 2
FP_SIZE = 1024


def _fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    return AllChem.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE).GetFingerprint(mol)


def _canon(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol) if mol is not None else None


def compute_inout_sim_from_detailed(detailed_path: Path) -> list[float]:
    """Tanimoto similarity between start and final SMILES for reacted episodes."""
    sims: list[float] = []
    for row in _iter_block_rows(detailed_path, "trajectories"):
        try:
            n_rx = int(row.get("num_reactions", "0"))
        except ValueError:
            n_rx = 0
        if n_rx < 1:
            continue
        start = str(row.get("start_smiles", "")).strip()
        out = str(row.get("final_smiles", "")).strip()
        if not start or not out:
            continue
        cs, co = _canon(start), _canon(out)
        if not cs or not co or cs == co:
            continue
        fa, fb = _fp(out), _fp(start)
        if fa is None or fb is None:
            continue
        sims.append(float(DataStructs.TanimotoSimilarity(fa, fb)))
    return sims
