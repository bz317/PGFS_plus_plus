"""Shared PGFS++ trainer helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import wandb

from src.config import project_root
from src.logging.wandb_metrics import define_ppo_compatible_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_dir(run_id: str) -> Path:
    path = project_root() / "runs" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def init_wandb(config: dict, algorithm: str, experiment_name: str):
    value = os.environ.get("WANDB_RESUME")
    if value is not None and value not in {"allow", "must", "never", "auto"}:
        os.environ.pop("WANDB_RESUME", None)

    # Training must work without a W&B login. Users who want logging can set
    # WANDB_API_KEY and/or WANDB_MODE=online.
    if not os.environ.get("WANDB_MODE") and not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_MODE"] = "disabled"

    project = os.getenv("WANDB_PROJECT", config.get("project", "PGFS++"))
    init_kw = {
        "project": project,
        "name": experiment_name,
        "job_type": f"train-{algorithm.lower()}",
        "save_code": True,
        "resume": "allow" if config.get("wandb_resume") else "never",
        "config": config,
    }
    entity = os.getenv("WANDB_ENTITY") or config.get("entity")
    if entity:
        init_kw["entity"] = entity
    run_id = (config.get("training") or {}).get("run_id")
    if config.get("wandb_resume") and run_id:
        init_kw["id"] = str(run_id)
        init_kw["resume"] = "must"
    run = wandb.init(**init_kw)
    define_ppo_compatible_metrics()
    return run
