#!/usr/bin/env python3
"""Verify that a PGFS++ checkout can import deps and load shipped data."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _ok(msg: str) -> None:
    print(f"[ok] {msg}", flush=True)


def _fail(msg: str) -> None:
    print(f"[fail] {msg}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def main() -> None:
    try:
        import numpy as np
        import torch
        import yaml  # noqa: F401
        from rdkit import Chem
        from rdkit.Chem import QED
    except Exception as exc:  # pragma: no cover
        _fail(f"core import failed: {exc}")

    _ok(f"python {sys.version.split()[0]}")
    _ok(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        _ok(f"gpu {torch.cuda.get_device_name(0)}")

    required = {
        "data/reactants_train.pkl": REPO / "data/reactants_train.pkl",
        "data/reactants_test.pkl": REPO / "data/reactants_test.pkl",
        "data/templates.pkl": REPO / "data/templates.pkl",
        "data/r2_valid_indices.npz": REPO / "data/r2_valid_indices.npz",
        "scoring/seh/bengio2021flow_proxy.pkl.gz": REPO
        / "scoring/seh/bengio2021flow_proxy.pkl.gz",
    }
    for label, path in required.items():
        if not path.is_file():
            _fail(f"missing {label}")
        _ok(f"found {label} ({path.stat().st_size} bytes)")

    with (REPO / "data/reactants_train.pkl").open("rb") as f:
        train = pickle.load(f)
    with (REPO / "data/templates.pkl").open("rb") as f:
        templates = pickle.load(f)
    if not train:
        _fail("reactants_train.pkl is empty")
    if not templates:
        _fail("templates.pkl is empty")

    smiles = next(iter(train.keys() if isinstance(train, dict) else train))
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        _fail(f"RDKit could not parse a training SMILES: {smiles!r}")
    qed = float(QED.qed(mol))
    _ok(f"QED({smiles[:48]}...) = {qed:.3f}")

    from src.chem.r2_valid_indices_store import try_load_r2_valid_indices_store
    from src.config import resolve_path

    store = try_load_r2_valid_indices_store(
        "data/r2_valid_indices.npz", resolve_path_fn=resolve_path
    )
    if store is None:
        _fail("could not load data/r2_valid_indices.npz")
    _ok(f"R2 mask store: {store.num_templates} templates")

    try:
        import torch_geometric
        import torch_sparse  # noqa: F401

        _ok(f"torch_geometric {torch_geometric.__version__} (ΔSEH ready)")
    except Exception as exc:
        print(
            f"[warn] PyTorch Geometric not available ({exc}). "
            "ΔQED training/eval is fine; ΔSEH needs the bootstrap PyG stack.",
            flush=True,
        )

    _ = np  # keep import used
    print("PGFS++ install check passed.", flush=True)


if __name__ == "__main__":
    main()
