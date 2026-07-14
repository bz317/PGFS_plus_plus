"""Checkpoint helpers for PGFS++ (PPO)."""

from __future__ import annotations

import re
from pathlib import Path

from src.config import resolve_path

_PPO_STEP_RE = re.compile(r"^model_step_(\d+)$")


def latest_ppo_checkpoint(run_dir: Path | str) -> Path | None:
    root = Path(resolve_path(str(run_dir)))
    if not root.is_dir():
        return None
    numbered: list[tuple[int, Path]] = []
    for path in root.glob("model_step_*.pt"):
        match = _PPO_STEP_RE.fullmatch(path.stem)
        if match:
            numbered.append((int(match.group(1)), path))
    if numbered:
        return max(numbered, key=lambda item: item[0])[1]
    for name in ("final_model.pt", "best_model.pt"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None
