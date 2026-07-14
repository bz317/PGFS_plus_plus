"""RDKit fingerprint helpers."""

from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def morgan_fp_array(smiles: str | None, *, fp_size: int = 1024, radius: int = 2) -> np.ndarray:
    """Return a binary Morgan fingerprint array, or zeros for invalid input."""
    arr = np.zeros((fp_size,), dtype=np.float32)
    if not smiles:
        return arr
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return arr
    fp = AllChem.GetMorganGenerator(radius=radius, fpSize=fp_size).GetFingerprint(mol)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32, copy=False)
