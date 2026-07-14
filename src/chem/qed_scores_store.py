"""Mmap-backed store of precomputed per-reactant QED scores for training curricula."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.chem.r2_valid_indices_store import sha256_pool_keys

ARTIFACT_VERSION = 1


class QedScoresStore:
    """Pool-aligned QED scores with sha256-validated pool binding."""

    __slots__ = ("qed", "order", "pool_size", "pool_keys_sha256")

    def __init__(
        self,
        *,
        qed: np.ndarray,
        order: np.ndarray,
        pool_size: int,
        pool_keys_sha256: str,
    ) -> None:
        self.qed = np.asarray(qed, dtype=np.float32)
        self.order = np.asarray(order, dtype=np.int64)
        self.pool_size = int(pool_size)
        self.pool_keys_sha256 = str(pool_keys_sha256)

    def validate_pool(self, pool_keys: list[str]) -> None:
        if len(pool_keys) != self.pool_size:
            raise ValueError(
                f"QED store pool_size={self.pool_size} but reactants pool has "
                f"{len(pool_keys)} entries"
            )
        digest = sha256_pool_keys(pool_keys)
        if digest != self.pool_keys_sha256:
            raise ValueError(
                "QED store pool_keys_sha256 does not match reactants pickle "
                "(pool key order may have changed)"
            )

    @classmethod
    def from_npz(cls, path: str | Path, *, mmap: bool = True) -> "QedScoresStore":
        path = Path(path)
        meta = _load_manifest(_manifest_path_for_npz(path))
        mode = "r" if mmap else None
        with np.load(path, mmap_mode=mode) as data:
            qed = np.asarray(data["qed"])
            order = np.asarray(data["order"]) if "order" in data else np.argsort(
                np.where(np.isnan(qed), np.inf, qed), kind="stable"
            )
        return cls(
            qed=qed,
            order=order,
            pool_size=int(meta["pool_size"]),
            pool_keys_sha256=str(meta["pool_keys_sha256"]),
        )


def _normalized_optional_path(raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"none", "null", "~"}:
        return None
    return Path(text)


def try_load_qed_scores_store(
    raw_path: str | Path | None,
    *,
    resolve_path_fn,
    mmap: bool = True,
) -> QedScoresStore | None:
    """Load precomputed QED scores when configured and present, else ``None``.

    Returns ``None`` (curriculum falls back to on-the-fly QED) when the config
    key is empty or the file is missing, so configs without the key are
    unaffected.
    """
    path = _normalized_optional_path(raw_path)
    if path is None:
        return None
    path = Path(resolve_path_fn(str(path)))
    if not path.is_file():
        print(
            f"[curriculum] qed_scores_file={path} not found; "
            "computing QED on the fly.",
            flush=True,
        )
        return None
    store = QedScoresStore.from_npz(path, mmap=mmap)
    print(
        f"[curriculum] loaded precomputed QED scores from {path} "
        f"(pool_size={store.pool_size})",
        flush=True,
    )
    return store


def compute_order(qed: np.ndarray) -> np.ndarray:
    """Ascending QED order (lowest QED first); NaN sorts last. Stable."""
    qed = np.asarray(qed, dtype=np.float64)
    keys = np.where(np.isnan(qed), np.inf, qed)
    return np.argsort(keys, kind="stable").astype(np.int64)


def save_npz(
    path: str | Path,
    *,
    qed: np.ndarray,
    pool_keys_sha256: str,
) -> None:
    """Write uncompressed ``qed`` + ``order`` arrays for mmap-friendly loads."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    qed = np.asarray(qed, dtype=np.float32)
    np.savez(path, qed=qed, order=compute_order(qed))


def _manifest_path_for_npz(npz_path: Path) -> Path:
    stem = npz_path.name
    if stem.endswith(".npz"):
        stem = stem[: -len(".npz")]
    return npz_path.with_name(f"{stem}_manifest.json")


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing manifest {path}. Re-run precompute_qed_scores.py."
        )
    return json.loads(path.read_text(encoding="utf-8"))
