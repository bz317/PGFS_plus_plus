"""Pluggable molecule representations for state input *and* R(2) action space.

This module backs the Bi-TD3 (PGFS-style) representation knobs:

* ``env.molecule_representation`` — sets BOTH R(1) state and R(2) action space
  (default for all algorithms).
* ``env.state_representation`` / ``env.r2_representation`` — PGFS-only split
  (paper: ECFP state + RLV2 action).

* ``"morgan"`` (alias ``"ecfp"`` / ``"ecfp4"``) — 1024-d Morgan FP, radius 2.
  Binary {0, 1}^1024. Identical to ``chem.fingerprints.morgan_fp_array`` so
  existing PPO/A2C/GraphTrans pipelines stay bit-equivalent (this is the
  default everywhere).
* ``"maccs"`` — 167-d MACCS keys (RDKit returns 167 bits incl. unused bit 0).
  Binary {0, 1}^167.
* ``"rlv2"`` (alias ``"moldset"``) — 35-d normalised RDKit descriptor set from
  PGFS Appendix A (Gottipati et al. 2020). Continuous ~[-1, +1]^35 after a
  per-feature z-score + ±3σ clip + ÷3 rescale fit on the **training**
  reactant pool. Stats are cached next to the training pickle.

Each call to :func:`make_representation` returns a :class:`Representation`
bundle of ``(name, fn, dim, is_binary)``. The binary flag lets downstream
PGFS code (KNN FAISS index, replay buffer warm-up rescale) treat binary
keys symmetrically (rescale to {-1, +1}) while leaving continuous
descriptors untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys  # noqa: F401  (AllChem warms Morgan)

from src.chem.fingerprints import morgan_fp_array

MORGAN_DIM = 1024
MACCS_DIM = 167

# PGFS Appendix A — RLV2 / MolDSet (35 RDKit descriptors).
RLV2_FEATURE_NAMES: tuple[str, ...] = (
    "MaxEStateIndex",
    "MinEStateIndex",
    "MinAbsEStateIndex",
    "qed",
    "MolWt",
    "FpDensityMorgan1",
    "BalabanJ",
    "PEOE_VSA10",
    "PEOE_VSA11",
    "PEOE_VSA6",
    "PEOE_VSA7",
    "PEOE_VSA8",
    "PEOE_VSA9",
    "SMR_VSA7",
    "SlogP_VSA3",
    "SlogP_VSA5",
    "EState_VSA2",
    "EState_VSA3",
    "EState_VSA4",
    "EState_VSA5",
    "EState_VSA6",
    "FractionCSP3",
    "MolLogP",
    "Kappa2",
    "PEOE_VSA2",
    "SMR_VSA5",
    "SMR_VSA6",
    "EState_VSA7",
    "Chi4v",
    "SMR_VSA10",
    "SlogP_VSA4",
    "SlogP_VSA6",
    "EState_VSA8",
    "EState_VSA9",
    "VSA_EState9",
)
RLV2_DIM = len(RLV2_FEATURE_NAMES)


def _rlv2_calculator():
    # Local import: ``MoleculeDescriptors`` is a private rdkit submodule.
    from rdkit.ML.Descriptors import MoleculeDescriptors

    return MoleculeDescriptors.MolecularDescriptorCalculator(list(RLV2_FEATURE_NAMES))


def compute_morgan(smiles: str | None) -> np.ndarray:
    return morgan_fp_array(smiles)


def compute_maccs(smiles: str | None) -> np.ndarray:
    arr = np.zeros((MACCS_DIM,), dtype=np.float32)
    if not smiles:
        return arr
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return arr
    fp = MACCSkeys.GenMACCSKeys(mol)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32, copy=False)


def compute_rlv2_raw(smiles: str | None, calc=None) -> np.ndarray:
    """Compute the raw, **unnormalised** 35-d RLV2 vector for ``smiles``.

    Returns a NaN-filled vector for invalid/empty SMILES so the caller can
    decide how to handle them during fitting.
    """
    arr = np.full((RLV2_DIM,), np.nan, dtype=np.float64)
    if not smiles:
        return arr
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return arr
    calc = calc or _rlv2_calculator()
    values = calc.CalcDescriptors(mol)
    arr[:] = values
    return arr


@dataclass
class RLV2Normalizer:
    """Per-feature z-score with ±``clip``σ clipping and ÷``clip`` rescaling.

    Output ∈ [-1, +1]^35 (modulo numerical edge cases), matching the actor's
    tanh range exactly so the PGFS KNN query and FAISS keys share a single
    symmetric space.
    """

    mean: np.ndarray  # (35,)
    std: np.ndarray  # (35,)
    clip: float = 3.0

    def transform(self, raw: np.ndarray) -> np.ndarray:
        x = np.where(np.isfinite(raw), raw, self.mean)
        x = (x - self.mean) / self.std
        x = np.clip(x, -self.clip, self.clip) / self.clip
        return x.astype(np.float32, copy=False)

    def transform_smiles(self, smiles: str | None, calc=None) -> np.ndarray:
        return self.transform(compute_rlv2_raw(smiles, calc=calc))

    def save(self, path: Path) -> None:
        np.savez(
            path,
            mean=self.mean,
            std=self.std,
            clip=np.asarray(self.clip),
            feature_names=np.asarray(RLV2_FEATURE_NAMES),
        )

    @classmethod
    def load(cls, path: Path) -> "RLV2Normalizer":
        d = np.load(path, allow_pickle=False)
        return cls(mean=d["mean"], std=d["std"], clip=float(d["clip"]))

    @classmethod
    def fit(cls, smiles_list: Sequence[str], clip: float = 3.0) -> "RLV2Normalizer":
        calc = _rlv2_calculator()
        raw = np.stack([compute_rlv2_raw(s, calc=calc) for s in smiles_list])
        raw = np.where(np.isfinite(raw), raw, np.nan)
        mean = np.nanmean(raw, axis=0)
        std = np.nanstd(raw, axis=0)
        # Replace pathological columns (all-NaN or zero variance) with safe
        # defaults so divide-by-zero never reaches transform().
        std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        return cls(mean=mean.astype(np.float64), std=std.astype(np.float64), clip=float(clip))


def _norm_cache_path(training_file: str | Path) -> Path:
    p = Path(training_file)
    return p.with_suffix(p.suffix + ".rlv2_norm.npz")


def load_or_fit_rlv2_normalizer(
    *,
    training_file: str | Path | None = None,
    training_smiles: Sequence[str] | None = None,
    clip: float = 3.0,
) -> RLV2Normalizer:
    """Load cached RLV2 stats from ``<training_file>.rlv2_norm.npz`` or fit
    them from ``training_smiles`` and persist to disk.

    The cache key is the path itself — train and eval envs share stats as
    long as both pass the same ``training_file`` (this is what
    ``env_kwargs`` does).
    """
    cache: Path | None = None
    if training_file is not None:
        cache = _norm_cache_path(training_file)
        if cache.exists():
            return RLV2Normalizer.load(cache)
    if training_smiles is None:
        raise ValueError(
            "RLV2 normalisation: cache miss and no ``training_smiles`` to fit."
        )
    norm = RLV2Normalizer.fit(training_smiles, clip=clip)
    if cache is not None:
        try:
            norm.save(cache)
        except OSError:
            # Read-only filesystem etc. — fall back to in-memory stats.
            pass
    return norm


@dataclass
class Representation:
    """Bundles a SMILES → vector callable with its metadata."""

    name: str
    fn: Callable[[str | None], np.ndarray]
    dim: int
    is_binary: bool


SUPPORTED_REPRESENTATIONS = ("morgan", "ecfp", "ecfp4", "maccs", "rlv2", "moldset")


def make_representation(
    name: str,
    *,
    training_smiles: Sequence[str] | None = None,
    training_file: str | Path | None = None,
) -> Representation:
    """Build a :class:`Representation` for ``name``.

    For ``"rlv2"``, supply either a cached ``training_file`` or a list of
    ``training_smiles`` (the env will pass both when available so the cache
    can be (re)built on first run).
    """
    n = (name or "").strip().lower()
    if n in {"morgan", "ecfp", "ecfp4"}:
        return Representation("morgan", compute_morgan, MORGAN_DIM, is_binary=True)
    if n == "maccs":
        return Representation("maccs", compute_maccs, MACCS_DIM, is_binary=True)
    if n in {"rlv2", "moldset"}:
        norm = load_or_fit_rlv2_normalizer(
            training_file=training_file,
            training_smiles=training_smiles,
        )
        calc = _rlv2_calculator()
        # Capture ``calc`` so we don't rebuild it per molecule (saves ~20% on
        # the env-init pool encode).
        def _rlv2_fn(smiles: str | None) -> np.ndarray:
            return norm.transform_smiles(smiles, calc=calc)

        return Representation("rlv2", _rlv2_fn, RLV2_DIM, is_binary=False)
    raise ValueError(
        f"Unknown molecule_representation {name!r}. "
        f"Supported: 'morgan' / 'ecfp', 'maccs', 'rlv2' / 'moldset'."
    )


__all__ = [
    "MACCS_DIM",
    "MORGAN_DIM",
    "RLV2_DIM",
    "RLV2_FEATURE_NAMES",
    "RLV2Normalizer",
    "Representation",
    "SUPPORTED_REPRESENTATIONS",
    "compute_maccs",
    "compute_morgan",
    "compute_rlv2_raw",
    "load_or_fit_rlv2_normalizer",
    "make_representation",
]
