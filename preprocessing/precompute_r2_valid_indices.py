#!/usr/bin/env python3
"""Precompute pattern-valid R2 pool indices for each bimolecular template.

Example: python preprocessing/precompute_r2_valid_indices.py --jobs 16
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chem.r2_valid_indices_store import (  # noqa: E402
    ARTIFACT_VERSION,
    EQUIVALENT_MASKING,
    R2_MASK_KIND_PATTERN,
    pack_csr,
    save_npz,
    sha256_pool_keys,
    unpack_csr,
)
from src.chem.reaction_manager import BI_TYPE, UNI_TYPES, ReactionManager  # noqa: E402

DEFAULT_REACTANTS = (ROOT / "data/reactants_train.pkl").resolve()
DEFAULT_TEMPLATES = (ROOT / "data/templates.pkl").resolve()
DEFAULT_OUTPUT = (ROOT / "data/r2_valid_indices.npz").resolve()
DEFAULT_MANIFEST = (ROOT / "data/r2_valid_indices_manifest.json").resolve()
DEFAULT_SMOKE_TEMPLATE = 2  # first bimolecular template under ReactionManager indexing


@dataclass(frozen=True)
class TemplateR2Query:
    template_index: int
    template_type: str
    r2_queries: tuple[Any, ...]  # OR-ed RDKit query mols; empty for uni / invalid bi


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_templates(path: Path) -> dict[int, dict]:
    obj = pickle.load(path.open("rb"))
    if isinstance(obj, dict):
        # Match ReactionManager._normalize_templates: insertion order -> 0..N-1
        return {i: dict(t) for i, (_, t) in enumerate(obj.items())}
    return {i: dict(t) for i, t in enumerate(obj)}


def _load_pool_keys(path: Path, *, max_pool: int | None) -> list[str]:
    obj = pickle.load(path.open("rb"))
    if isinstance(obj, dict):
        keys = list(obj.keys())
    else:
        keys = list(obj)
    if max_pool is not None:
        keys = keys[: int(max_pool)]
    return keys


def _compile_r2_queries(templates: dict[int, dict]) -> list[TemplateR2Query]:
    out: list[TemplateR2Query] = []
    for idx in sorted(templates):
        template = templates[idx]
        ttype = template.get("type", "unimolecular")
        if ttype in UNI_TYPES or ttype != BI_TYPE:
            out.append(TemplateR2Query(idx, ttype, ()))
            continue
        queries = tuple(ReactionManager.r2_pattern_queries(template))
        out.append(TemplateR2Query(idx, ttype, queries))
    return out


def _scan_pool_for_queries(pool_keys: list[str], r2_queries: tuple[Any, ...]) -> np.ndarray:
    if not r2_queries:
        return np.zeros(0, dtype=np.int32)
    hits: list[int] = []
    for i, smi in enumerate(pool_keys):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if any(mol.HasSubstructMatch(q, useChirality=True) for q in r2_queries):
            hits.append(i)
    return np.asarray(hits, dtype=np.int32)


def _scan_one_template(
    template_index: int,
    template_type: str,
    pool_keys: list[str],
    r2_queries: tuple[Any, ...],
) -> tuple[int, np.ndarray]:
    if template_type in UNI_TYPES or not r2_queries:
        return template_index, np.zeros(0, dtype=np.int32)
    return template_index, _scan_pool_for_queries(pool_keys, r2_queries)


_POOL_KEYS: list[str] | None = None


def _init_pool_worker(pool_keys: list[str]) -> None:
    global _POOL_KEYS
    _POOL_KEYS = pool_keys


def _worker_scan(args: tuple[int, str, tuple[Any, ...]]) -> tuple[int, np.ndarray]:
    if _POOL_KEYS is None:
        raise RuntimeError("worker pool keys not initialized")
    template_index, template_type, r2_queries = args
    return _scan_one_template(template_index, template_type, _POOL_KEYS, r2_queries)


def _select_template_ids(
    queries: list[TemplateR2Query],
    *,
    template_ids: list[int] | None,
    smoke: bool,
) -> list[int]:
    if smoke:
        return [DEFAULT_SMOKE_TEMPLATE]
    if template_ids is not None:
        return sorted(set(template_ids))
    return [q.template_index for q in queries]


def _load_existing_indices(path: Path, num_templates: int) -> dict[int, np.ndarray]:
    if not path.is_file():
        return {i: np.zeros(0, dtype=np.int32) for i in range(num_templates)}
    if path.suffix == ".npz":
        with np.load(path, mmap_mode="r") as data:
            return unpack_csr(np.asarray(data["indptr"]), np.asarray(data["indices"]))
    obj = pickle.load(path.open("rb"))
    if "indices_by_template" in obj:
        return {
            int(k): np.asarray(v, dtype=np.int32)
            for k, v in obj["indices_by_template"].items()
        }
    return unpack_csr(np.asarray(obj["indptr"]), np.asarray(obj["indices"]))


def pick_default_verify_template_ids(manifest: dict[str, Any], *, n: int = 10) -> list[int]:
    """Pick a small stratified subset of bimolecular templates for parity checks."""
    per = manifest.get("per_template", {})
    bi = sorted(
        (int(k), int(v.get("n_valid_r2", 0)))
        for k, v in per.items()
        if v.get("type") == BI_TYPE
    )
    if not bi:
        return []
    if len(bi) <= n:
        return [t for t, _ in bi]

    counts = {t: c for t, c in bi}
    bi_ids = [t for t, _ in bi]
    anchors: list[int] = []
    heavy = max(counts, key=counts.get)
    anchors.append(heavy)
    nonzero = [t for t in bi_ids if counts[t] > 0]
    if nonzero:
        anchors.append(min(nonzero, key=lambda t: counts[t]))
    zero = [t for t in bi_ids if counts[t] == 0]
    if zero:
        anchors.append(zero[0])

    picks = list(dict.fromkeys(anchors))
    remaining = [t for t in bi_ids if t not in picks]
    slots = max(0, n - len(picks))
    if slots and remaining:
        idxs = np.linspace(0, len(remaining) - 1, slots).round().astype(int)
        picks.extend(remaining[int(i)] for i in idxs)
    return sorted(dict.fromkeys(picks))[:n]


def verify_templates_against_reaction_manager(
    *,
    templates: dict[int, dict],
    pool_keys: list[str],
    indices_by_template: dict[int, np.ndarray],
    template_ids: list[int],
    sample_pool: int | None = None,
    spot_check_n: int = 5,
) -> dict[str, Any]:
    """Check precomputed R2 indices against ReactionManager (r2_available pattern path)."""
    pool_dict = {smi: None for smi in pool_keys}
    rm = ReactionManager(templates, pool_dict)
    check_keys = pool_keys if sample_pool is None else pool_keys[:sample_pool]

    checked: list[int] = []
    per_template: dict[str, Any] = {}
    t_all = time.perf_counter()

    for t_idx in template_ids:
        ttype = templates[t_idx].get("type", "unimolecular")
        if ttype in UNI_TYPES:
            per_template[str(t_idx)] = {"skipped": True, "reason": "unimolecular"}
            continue

        t0 = time.perf_counter()
        pre = indices_by_template[t_idx]
        rm_idx = rm.get_valid_reactant_indices(t_idx)
        elapsed_s = time.perf_counter() - t0
        ok = pre.shape == rm_idx.shape and np.array_equal(pre, rm_idx)
        per_template[str(t_idx)] = {
            "precompute_n": int(pre.size),
            "reaction_manager_n": int(rm_idx.size),
            "elapsed_seconds": round(elapsed_s, 3),
            "parity_ok": bool(ok),
        }
        checked.append(t_idx)
        status = "OK" if ok else "FAIL"
        print(
            f"[verify] template {t_idx:3d}: {status} "
            f"precompute={pre.size} reaction_manager={rm_idx.size} "
            f"({elapsed_s:.1f}s)",
            flush=True,
        )
        if not ok:
            raise AssertionError(
                f"parity failed for template {t_idx}: "
                f"precompute={pre.size} reaction_manager={rm_idx.size}"
            )

        if pre.size == 0:
            continue

        template = templates[t_idx]
        for global_i in pre[: min(spot_check_n, pre.size)]:
            smi = check_keys[int(global_i)]
            if not rm.match_template(smi, template)["second"]:
                raise AssertionError(
                    f"index {global_i} ({smi!r}) fails match_template second for T={t_idx}"
                )

    elapsed_total = time.perf_counter() - t_all
    print(
        f"[verify] ReactionManager r2_available parity OK for templates {checked} "
        f"(pool_keys checked={len(check_keys)}, {elapsed_total:.1f}s total)",
        flush=True,
    )
    return {
        "masking": EQUIVALENT_MASKING,
        "reaction_manager_path": "get_valid_reactant_indices (r2_available pattern set)",
        "pool_keys_checked": len(check_keys),
        "template_ids_requested": list(template_ids),
        "template_ids_checked": checked,
        "elapsed_seconds": round(elapsed_total, 3),
        "per_template": per_template,
        "status": "ok",
    }


def _verify_against_reaction_manager(
    *,
    templates: dict[int, dict],
    pool_keys: list[str],
    indices_by_template: dict[int, np.ndarray],
    template_ids: list[int],
    sample_pool: int | None,
) -> None:
    verify_templates_against_reaction_manager(
        templates=templates,
        pool_keys=pool_keys,
        indices_by_template=indices_by_template,
        template_ids=template_ids,
        sample_pool=sample_pool,
    )


def _build_manifest(
    *,
    args: argparse.Namespace,
    pool_keys: list[str],
    templates: dict[int, dict],
    indices_by_template: dict[int, np.ndarray],
    template_ids_built: list[int],
    elapsed_s: float,
) -> dict[str, Any]:
    per_template = {}
    total_indices = 0
    for t_idx in sorted(indices_by_template):
        arr = indices_by_template[t_idx]
        n = int(arr.size)
        total_indices += n
        if n > 0 or t_idx in template_ids_built:
            per_template[str(t_idx)] = {
                "type": templates[t_idx].get("type", "unimolecular"),
                "n_valid_r2": n,
            }

    bi_with_zero = sum(
        1
        for t_idx, t in templates.items()
        if t.get("type") == BI_TYPE and indices_by_template[t_idx].size == 0
    )

    return {
        "generated_at_utc": _utc_now(),
        "artifact_version": ARTIFACT_VERSION,
        "reactants_pkl": str(args.reactants_pkl.resolve()),
        "templates_pkl": str(args.templates_pkl.resolve()),
        "output_npz": str(args.output.resolve()),
        "pool_size": len(pool_keys),
        "pool_keys_sha256": sha256_pool_keys(pool_keys),
        "storage_format": "csr_npz",
        "npz_arrays": ["indptr", "indices"],
        "r2_mask_kind": R2_MASK_KIND_PATTERN,
        "equivalent_masking": EQUIVALENT_MASKING,
        "uses_apply_reaction": False,
        "num_templates": len(templates),
        "templates_built": template_ids_built,
        "bimolecular_templates": sum(1 for t in templates.values() if t.get("type") == BI_TYPE),
        "bimolecular_with_zero_r2": bi_with_zero,
        "total_stored_indices": total_indices,
        "elapsed_seconds": round(elapsed_s, 3),
        "per_template": per_template,
        "notes": (
            "indices_by_template[t] are global indices into list(reactants_train.pkl.keys()). "
            "Uni templates are stored as empty arrays. Pattern-only (r2_available); "
            "no apply_reaction validation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reactants-pkl", type=Path, default=DEFAULT_REACTANTS)
    parser.add_argument("--templates-pkl", type=Path, default=DEFAULT_TEMPLATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--template-ids",
        type=int,
        nargs="*",
        default=None,
        help="Subset of template indices to (re)build. Default: all templates.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Build only template {DEFAULT_SMOKE_TEMPLATE} (first bi template).",
    )
    parser.add_argument(
        "--max-pool",
        type=int,
        default=None,
        help="Optional cap on pool size (first N reactants) for quick dev runs.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel workers across templates (recommended for full build).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check parity against ReactionManager.r2_mask after build.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification even in --smoke mode.",
    )
    args = parser.parse_args()

    if not args.reactants_pkl.is_file():
        raise FileNotFoundError(args.reactants_pkl)
    if not args.templates_pkl.is_file():
        raise FileNotFoundError(args.templates_pkl)

    t0 = time.perf_counter()
    templates = _load_templates(args.templates_pkl)
    pool_keys = _load_pool_keys(args.reactants_pkl, max_pool=args.max_pool)
    queries = _compile_r2_queries(templates)
    build_ids = _select_template_ids(queries, template_ids=args.template_ids, smoke=args.smoke)

    print(
        f"[precompute] pool={len(pool_keys)} templates={len(templates)} "
        f"build_ids={build_ids} jobs={args.jobs}",
        flush=True,
    )

    pool_digest = sha256_pool_keys(pool_keys)
    indices_by_template: dict[int, np.ndarray] = {
        i: np.zeros(0, dtype=np.int32) for i in templates
    }
    if args.output.is_file() and (args.template_ids is not None or args.smoke):
        prev_map = _load_existing_indices(args.output, len(templates))
        manifest_path = args.manifest
        if manifest_path.is_file():
            prev_digest = json.loads(manifest_path.read_text(encoding="utf-8")).get(
                "pool_keys_sha256"
            )
            if prev_digest and prev_digest != pool_digest:
                raise ValueError(
                    "Existing artifact pool_keys_sha256 does not match current reactants pkl."
                )
        for t_idx, arr in prev_map.items():
            indices_by_template[int(t_idx)] = np.asarray(arr, dtype=np.int32)

    query_by_id = {q.template_index: q for q in queries}
    serial_tasks = [
        (
            t_idx,
            query_by_id[t_idx].template_type,
            pool_keys,
            query_by_id[t_idx].r2_queries,
        )
        for t_idx in build_ids
    ]
    parallel_tasks = [
        (t_idx, query_by_id[t_idx].template_type, query_by_id[t_idx].r2_queries)
        for t_idx in build_ids
    ]

    if args.jobs <= 1:
        for task in serial_tasks:
            t_idx, arr = _scan_one_template(*task)
            indices_by_template[t_idx] = arr
            print(
                f"  template {t_idx:3d} ({templates[t_idx].get('type', '?')}): "
                f"{arr.size} valid R2",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(
            max_workers=args.jobs,
            initializer=_init_pool_worker,
            initargs=(pool_keys,),
        ) as ex:
            futures = {ex.submit(_worker_scan, task): task[0] for task in parallel_tasks}
            for fut in as_completed(futures):
                t_idx, arr = fut.result()
                indices_by_template[t_idx] = arr
                print(
                    f"  template {t_idx:3d} ({templates[t_idx].get('type', '?')}): "
                    f"{arr.size} valid R2",
                    flush=True,
                )

    elapsed = time.perf_counter() - t0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_npz(
        args.output,
        indices_by_template=indices_by_template,
        num_templates=len(templates),
        pool_size=len(pool_keys),
        pool_keys_sha256=pool_digest,
    )

    manifest = _build_manifest(
        args=args,
        pool_keys=pool_keys,
        templates=templates,
        indices_by_template=indices_by_template,
        template_ids_built=build_ids,
        elapsed_s=elapsed,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    do_verify = args.verify or (args.smoke and not args.no_verify)
    if do_verify:
        _verify_against_reaction_manager(
            templates=templates,
            pool_keys=pool_keys,
            indices_by_template=indices_by_template,
            template_ids=build_ids,
            sample_pool=args.max_pool,
        )

    print(
        f"[done] wrote {args.output} ({args.output.stat().st_size / 1e6:.2f} MB, "
        f"CSR mmap) and {args.manifest} in {elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
