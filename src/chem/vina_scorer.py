"""Cached Vina reward helper for delta_vina training."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

from src.config import resolve_path


@dataclass
class VinaScorerConfig:
    target: str = "kras"
    vina_path: str | None = None
    opencl_binary_path: str | None = None
    reward_scale_max: float = -1.0
    reward_scale_min: float = -10.0
    thread: int = 8000
    mock: bool = False
    obabel_path: str | None = None


class MockQuickVina2GPU:
    """Deterministic pseudo-Vina for unit tests when GPU binary is unavailable."""

    def __init__(self, *, target: str, reward_scale_max: float, reward_scale_min: float, **_: Any) -> None:
        self.target_key = str(target).lower()
        self.reward_scale_max = float(reward_scale_max)
        self.reward_scale_min = float(reward_scale_min)

    def _pseudo_affinity(self, smiles: str) -> float:
        digest = hashlib.sha1(f"{self.target_key}:{smiles}".encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        # Affinities in [-10, -1]; lower (more negative) is better binding.
        return -10.0 + 9.0 * bucket

    def calculate_rewards(self, smiles: list[str]) -> tuple[list[str], list[float], list[float]]:
        affinities = [self._pseudo_affinity(s) for s in smiles]
        rewards = [
            (a + self.reward_scale_min) / (self.reward_scale_min + self.reward_scale_max) - 1.0
            for a in affinities
        ]
        return list(smiles), affinities, rewards


class VinaScorer:
    """SMILES-level cache in front of QuickVina2-GPU (or mock backend)."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self._reward_cache: dict[str, float] = {}
        self._affinity_cache: dict[str, float] = {}
        self.dock_calls = 0
        self.cache_hits = 0

    @classmethod
    def from_config(cls, cfg: dict | VinaScorerConfig | None) -> "VinaScorer":
        if cfg is None:
            cfg = {}
        if isinstance(cfg, VinaScorerConfig):
            parsed = cfg
        else:
            parsed = VinaScorerConfig(
                target=str(cfg.get("target", "kras")),
                vina_path=cfg.get("vina_path"),
                opencl_binary_path=cfg.get("opencl_binary_path"),
                reward_scale_max=float(cfg.get("reward_scale_max", -1.0)),
                reward_scale_min=float(cfg.get("reward_scale_min", -10.0)),
                thread=int(cfg.get("thread", 8000)),
                mock=bool(cfg.get("mock", False)),
                obabel_path=cfg.get("obabel_path"),
            )
        use_mock = parsed.mock or os.environ.get("GENMOLRL_MOCK_VINA", "").lower() in {
            "1",
            "true",
            "yes",
        }
        if use_mock:
            backend = MockQuickVina2GPU(
                target=parsed.target,
                reward_scale_max=parsed.reward_scale_max,
                reward_scale_min=parsed.reward_scale_min,
            )
        else:
            from src.chem.gpu_vina import QuickVina2GPU, gpu_vina_installed

            vina_path = resolve_path(parsed.vina_path) if parsed.vina_path else None
            opencl = (
                resolve_path(parsed.opencl_binary_path)
                if parsed.opencl_binary_path
                else None
            )
            if vina_path and not gpu_vina_installed(vina_path):
                raise FileNotFoundError(f"Vina executable not found: {vina_path}")
            backend = QuickVina2GPU(
                target=parsed.target,
                vina_path=vina_path,
                opencl_binary_path=opencl,
                reward_scale_max=parsed.reward_scale_max,
                reward_scale_min=parsed.reward_scale_min,
                thread=parsed.thread,
                obabel_path=parsed.obabel_path,
            )
        return cls(backend)

    def affinity(self, smiles: str | None) -> float:
        if not smiles:
            return 0.0
        cached = self._affinity_cache.get(smiles)
        if cached is not None:
            self.cache_hits += 1
            return cached
        if smiles in self._reward_cache:
            self.cache_hits += 1
            return self._affinity_cache.setdefault(smiles, 0.0)
        self.dock_calls += 1
        _, affinities, rewards = self._backend.calculate_rewards([smiles])
        aff = float(affinities[0]) if affinities else 0.0
        rew = float(rewards[0]) if rewards else 0.0
        self._affinity_cache[smiles] = aff
        self._reward_cache[smiles] = rew
        return aff

    def reward(self, smiles: str | None) -> float:
        if not smiles:
            return 0.0
        cached = self._reward_cache.get(smiles)
        if cached is not None:
            self.cache_hits += 1
            return cached
        if smiles in self._affinity_cache:
            self.cache_hits += 1
            return self._reward_cache.setdefault(smiles, 0.0)
        self.dock_calls += 1
        _, affinities, rewards = self._backend.calculate_rewards([smiles])
        aff = float(affinities[0]) if affinities else 0.0
        rew = float(rewards[0]) if rewards else 0.0
        self._affinity_cache[smiles] = aff
        self._reward_cache[smiles] = rew
        return rew

    def step_delta(self, previous_smiles: str | None, current_smiles: str | None) -> float:
        if not current_smiles:
            return 0.0
        curr = self.reward(current_smiles)
        prev = self.reward(previous_smiles) if previous_smiles else 0.0
        return float(curr - prev)

    def score_batch(self, smiles: list[str]) -> None:
        """Pre-score uncached SMILES in one QuickVina batch."""
        missing = [s for s in smiles if s and s not in self._reward_cache]
        if not missing:
            return
        self.dock_calls += 1
        _, affinities, rewards = self._backend.calculate_rewards(missing)
        for smi, aff, rew in zip(missing, affinities, rewards):
            self._affinity_cache[smi] = float(aff)
            self._reward_cache[smi] = float(rew)
