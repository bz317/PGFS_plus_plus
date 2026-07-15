#!/usr/bin/env python3
"""Build an anonymized PGFS++ supplementary code bundle."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEST = REPO.parent / "PGFS++_AAAI_code_supp"

REPLACEMENTS = {
    "ymhrz9yg": "multiplicative_bonus_model",
    "9gj82ve1": "multiplicative_seh_bonus_model",
    "4s_delta_qed_ymhrz9yg_1m_compact": "4s_delta_qed_multiplicative_bonus_model_1m_compact",
    "4s_delta_seh_9gj82ve1_compact": "4s_delta_seh_multiplicative_seh_bonus_model_compact",
}

EXCLUDE = {
    ".git",
    "LICENSE",
    "__pycache__",
    ".wandb",
    "wandb",
    "tools/build_aaai_code_supp.py",
}


def _rsync_copy() -> None:
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    cmd = [
        "rsync",
        "-a",
        f"{REPO}/",
        f"{DEST}/",
        *[f"--exclude={name}" for name in sorted(EXCLUDE)],
    ]
    subprocess.run(cmd, check=True)


def _replace_in_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    original = text
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)
    if text != original:
        path.write_text(text, encoding="utf-8")


def _walk_text_files() -> list[Path]:
    files: list[Path] = []
    for path in DEST.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix in {".py", ".sh", ".md", ".json", ".txt", ".yaml", ".yml", ".toml"}:
            files.append(path)
    return files


def _rename_paths() -> None:
    renames = [
        (DEST / "runs" / "ymhrz9yg", DEST / "runs" / "multiplicative_bonus_model"),
        (DEST / "runs" / "9gj82ve1", DEST / "runs" / "multiplicative_seh_bonus_model"),
        (
            DEST
            / "run_detailed_results/compact/4s_delta_qed_ymhrz9yg_1m_compact_results.txt",
            DEST
            / "run_detailed_results/compact/4s_delta_qed_multiplicative_bonus_model_1m_compact_results.txt",
        ),
        (
            DEST
            / "run_detailed_results/compact/4s_delta_qed_ymhrz9yg_1m_compact_results.summary.json",
            DEST
            / "run_detailed_results/compact/4s_delta_qed_multiplicative_bonus_model_1m_compact_results.summary.json",
        ),
        (
            DEST
            / "run_detailed_results/compact/4s_delta_seh_9gj82ve1_compact_results.txt",
            DEST
            / "run_detailed_results/compact/4s_delta_seh_multiplicative_seh_bonus_model_compact_results.txt",
        ),
        (
            DEST
            / "run_detailed_results/compact/4s_delta_seh_9gj82ve1_compact_results.summary.json",
            DEST
            / "run_detailed_results/compact/4s_delta_seh_multiplicative_seh_bonus_model_compact_results.summary.json",
        ),
    ]
    for src, dst in renames:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)


def _patch_gitignore() -> None:
    path = DEST / ".gitignore"
    if not path.exists():
        return
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip() != "runs/"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _patch_readme() -> None:
    path = DEST / "README.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "Official implementation of **PGFS++**",
        "**PGFS++** supplementary code",
    )
    text = text.replace("| Run ID | Reward | Checkpoint |", "| Model | Reward | Checkpoint |")
    text = text.replace(
        "export WANDB_MODE=disabled   # or set WANDB_PROJECT / WANDB_ENTITY",
        "export WANDB_MODE=disabled",
    )
    path.write_text(text, encoding="utf-8")


def _patch_runs_readme() -> None:
    path = DEST / "runs" / "README.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    text = text.replace("| Run ID |", "| Model |")
    path.write_text(text, encoding="utf-8")


def _scrub_residuals() -> None:
    patterns = [
        re.compile(r"/root/autodl-tmp[^\s\"']*"),
        re.compile(r"GenMolRL"),
        re.compile(r"data/Bi/"),
        re.compile(r"internal/ppo/"),
        re.compile(r"AAAI"),
        re.compile(r"Boqiao Zhang"),
        re.compile(r"bz317"),
        re.compile(r"cam\.ac\.uk"),
        re.compile(r"github\.com/bz317"),
    ]
    for path in _walk_text_files():
        text = path.read_text(encoding="utf-8")
        original = text
        for pattern in patterns:
            text = pattern.sub("<redacted>", text)
        if text != original:
            path.write_text(text, encoding="utf-8")


def main() -> None:
    _rsync_copy()
    _rename_paths()
    for path in _walk_text_files():
        _replace_in_file(path)
    _patch_gitignore()
    _patch_readme()
    _patch_runs_readme()
    _scrub_residuals()
    print(f"Built anonymized bundle at {DEST}")


if __name__ == "__main__":
    main()
