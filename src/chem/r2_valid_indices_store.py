"""Mmap-backed CSR store of pattern-valid R2 pool indices per template."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

ARTIFACT_VERSION = 1
# Precompute stores pattern-only valid R2 indices (``r2_available`` / ``substructure``
# R2 axis). It does NOT run ``apply_reaction`` (that is ``reaction_valid`` only).
R2_MASK_KIND_PATTERN = "pattern"
EQUIVALENT_MASKING = "r2_available"


def sha256_pool_keys(pool_keys: list[str]) -> str:
    h = hashlib.sha256()
    for smi in pool_keys:
        h.update(smi.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def pack_csr(indices_by_template: dict[int, np.ndarray], num_templates: int) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate per-template index arrays into CSR ``(indptr, indices)``."""
    indptr = np.zeros(num_templates + 1, dtype=np.int64)
    chunks: list[np.ndarray] = []
    for t in range(num_templates):
        arr = np.asarray(indices_by_template.get(t, ()), dtype=np.int32).ravel()
        chunks.append(arr)
        indptr[t + 1] = indptr[t] + arr.size
    indices = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int32)
    return indptr, indices


def unpack_csr(indptr: np.ndarray, indices: np.ndarray) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    n_templates = int(indptr.size) - 1
    for t in range(n_templates):
        sl = slice(int(indptr[t]), int(indptr[t + 1]))
        out[t] = np.asarray(indices[sl], dtype=np.int32)
    return out


class R2ValidIndicesStore:
    """CSR-backed per-template valid R2 global pool indices."""

    __slots__ = ("_indptr", "_indices", "num_templates", "pool_size", "pool_keys_sha256")

    def __init__(
        self,
        *,
        indptr: np.ndarray,
        indices: np.ndarray,
        pool_size: int,
        pool_keys_sha256: str,
    ) -> None:
        if indptr.ndim != 1 or indptr.size < 1:
            raise ValueError("indptr must be 1-D with length num_templates + 1")
        self._indptr = indptr
        self._indices = indices
        self.num_templates = int(indptr.size) - 1
        self.pool_size = int(pool_size)
        self.pool_keys_sha256 = str(pool_keys_sha256)

    def validate_pool(self, pool_keys: list[str]) -> None:
        if len(pool_keys) != self.pool_size:
            raise ValueError(
                f"R2 index store pool_size={self.pool_size} but reactants pool "
                f"has {len(pool_keys)} entries"
            )
        digest = sha256_pool_keys(pool_keys)
        if digest != self.pool_keys_sha256:
            raise ValueError(
                "R2 index store pool_keys_sha256 does not match reactants pickle "
                "(pool key order may have changed)"
            )

    def indices_for_template(self, template_index: int) -> np.ndarray:
        t = int(template_index)
        if t < 0 or t >= self.num_templates:
            raise IndexError(f"template_index {t} out of range [0, {self.num_templates})")
        sl = slice(int(self._indptr[t]), int(self._indptr[t + 1]))
        # Return a compact copy so callers may safely mutate / torch-convert.
        return np.asarray(self._indices[sl], dtype=np.int32)

    def has_any(self, template_index: int) -> bool:
        t = int(template_index)
        return int(self._indptr[t + 1]) > int(self._indptr[t])

    @classmethod
    def from_npz(cls, path: str | Path, *, mmap: bool = True) -> R2ValidIndicesStore:
        path = Path(path)
        manifest_path = _manifest_path_for_npz(path)
        meta = _load_manifest(manifest_path)

        mode = "r" if mmap else None
        with np.load(path, mmap_mode=mode) as data:
            indptr = np.asarray(data["indptr"])
            indices = np.asarray(data["indices"])
        return cls(
            indptr=indptr,
            indices=indices,
            pool_size=int(meta["pool_size"]),
            pool_keys_sha256=str(meta["pool_keys_sha256"]),
        )

    @classmethod
    def from_pickle(cls, path: str | Path) -> R2ValidIndicesStore:
        obj = pickle.load(Path(path).open("rb"))
        if "indptr" in obj and "indices" in obj:
            return cls(
                indptr=np.asarray(obj["indptr"]),
                indices=np.asarray(obj["indices"]),
                pool_size=int(obj["pool_size"]),
                pool_keys_sha256=str(obj["pool_keys_sha256"]),
            )
        indices_by_template = {
            int(k): np.asarray(v, dtype=np.int32)
            for k, v in obj["indices_by_template"].items()
        }
        num_templates = int(obj["num_templates"])
        indptr, indices = pack_csr(indices_by_template, num_templates)
        return cls(
            indptr=indptr,
            indices=indices,
            pool_size=int(obj["pool_size"]),
            pool_keys_sha256=str(obj["pool_keys_sha256"]),
        )

    @classmethod
    def load(cls, path: str | Path, *, mmap: bool = True) -> R2ValidIndicesStore:
        path = Path(path)
        if path.suffix == ".npz":
            return cls.from_npz(path, mmap=mmap)
        if path.suffix == ".pkl":
            return cls.from_pickle(path)
        raise ValueError(f"Unsupported R2 valid-indices artifact: {path}")


def _normalized_optional_path(raw: str | Path | None) -> Path | None:
    """Return a resolved path, or None when the config value is unset/empty."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"none", "null", "~"}:
        return None
    return Path(text)


def try_load_r2_valid_indices_store(
    raw_path: str | Path | None,
    *,
    resolve_path_fn,
    mmap: bool = True,
) -> R2ValidIndicesStore | None:
    """Load precomputed indices when ``raw_path`` is set and the artifact exists.

    Returns ``None`` (original on-the-fly R2 masking) when the config key is
    empty or the file is missing. Other SLURM jobs / Bi configs without this
    key therefore behave exactly as before.
    """
    path = _normalized_optional_path(raw_path)
    if path is None:
        return None
    path = Path(resolve_path_fn(str(path)))
    if not path.is_file():
        print(
            f"[ppo_bi] r2_valid_indices_file={path} not found; "
            "using on-the-fly pattern R2 masks.",
            flush=True,
        )
        return None
    store = R2ValidIndicesStore.load(path, mmap=mmap)
    print(
        f"[ppo_bi] loaded precomputed R2 valid indices from {path} "
        f"(masking={EQUIVALENT_MASKING}, pool_size={store.pool_size}, "
        f"templates={store.num_templates})",
        flush=True,
    )
    return store


def save_npz(
    path: str | Path,
    *,
    indices_by_template: dict[int, np.ndarray],
    num_templates: int,
    pool_size: int,
    pool_keys_sha256: str,
) -> None:
    """Write uncompressed CSR arrays for mmap-friendly training loads."""
    indptr, indices = pack_csr(indices_by_template, num_templates)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, indptr=indptr, indices=indices)


def _manifest_path_for_npz(npz_path: Path) -> Path:
    stem = npz_path.name
    if stem.endswith(".npz"):
        stem = stem[: -len(".npz")]
    return npz_path.with_name(f"{stem}_manifest.json")


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing manifest {path}. Re-run precompute_r2_valid_indices.py."
        )
    return json.loads(path.read_text(encoding="utf-8"))
