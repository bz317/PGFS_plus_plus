"""Cached sEH proxy scorer for delta_seh training."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

from src.config import resolve_path


@dataclass
class SehScorerConfig:
    weights_path: str | None = None
    device: str | None = None
    scale: float = 8.0
    mock: bool = False


class MockSehBackend:
    """Deterministic pseudo-SEH for unit tests."""

    def score_smiles(self, smiles: list[str]) -> list[float]:
        out: list[float] = []
        for smi in smiles:
            digest = hashlib.sha1(smi.encode("utf-8")).hexdigest()
            bucket = int(digest[:8], 16) / 0xFFFFFFFF
            out.append(0.01 + bucket * 2.0)
        return out


class SehProxyBackend:
    """Pretrained Bengio2021 sEH binding proxy (SynFlowNet-compatible)."""

    def __init__(self, *, weights_path: str | None, device: str | None, scale: float) -> None:
        import torch
        from rdkit import Chem

        from src.chem.seh_bengio2021flow import load_original_model, mol2graph

        self._Chem = Chem
        self._mol2graph = mol2graph
        self._scale = float(scale)
        self._device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        path = resolve_path(weights_path) if weights_path else None
        self._model = load_original_model(location=path)
        self._model.to(self._device).eval()

    def score_smiles(self, smiles: list[str]) -> list[float]:
        import torch
        from torch_geometric.data import Batch

        results = [0.0] * len(smiles)
        graphs = []
        indices: list[int] = []
        for idx, smi in enumerate(smiles):
            mol = self._Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                graph = self._mol2graph(mol)
            except (IndexError, ValueError):
                continue
            graphs.append(graph)
            indices.append(idx)
        if not graphs:
            return results
        batch = Batch.from_data_list(graphs).to(self._device)
        with torch.no_grad():
            preds = self._model(batch).reshape(-1) / self._scale
        preds = preds.clamp(1e-4, 100.0).detach().cpu().tolist()
        for idx, score in zip(indices, preds):
            results[idx] = float(score)
        return results


class SehScorer:
    """SMILES-level cache in front of the sEH proxy (or mock backend)."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self._cache: dict[str, float] = {}
        self.score_calls = 0
        self.cache_hits = 0

    @classmethod
    def from_config(cls, cfg: dict | SehScorerConfig | None) -> "SehScorer":
        if cfg is None:
            cfg = {}
        if isinstance(cfg, SehScorerConfig):
            parsed = cfg
        else:
            parsed = SehScorerConfig(
                weights_path=cfg.get("weights_path"),
                device=cfg.get("device"),
                scale=float(cfg.get("scale", 8.0)),
                mock=bool(cfg.get("mock", False)),
            )
        use_mock = parsed.mock or os.environ.get("GENMOLRL_MOCK_SEH", "").lower() in {
            "1",
            "true",
            "yes",
        }
        if use_mock:
            backend: Any = MockSehBackend()
        else:
            backend = SehProxyBackend(
                weights_path=parsed.weights_path,
                device=parsed.device,
                scale=parsed.scale,
            )
        return cls(backend)

    def reward(self, smiles: str | None) -> float:
        if not smiles:
            return 0.0
        cached = self._cache.get(smiles)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.score_calls += 1
        score = float(self._backend.score_smiles([smiles])[0])
        self._cache[smiles] = score
        return score

    def step_delta(self, previous_smiles: str | None, current_smiles: str | None) -> float:
        if not current_smiles:
            return 0.0
        curr = self.reward(current_smiles)
        prev = self.reward(previous_smiles) if previous_smiles else 0.0
        return float(curr - prev)

    def score_batch(self, smiles: list[str]) -> None:
        missing = [s for s in smiles if s and s not in self._cache]
        if not missing:
            return
        self.score_calls += 1
        scores = self._backend.score_smiles(missing)
        for smi, score in zip(missing, scores):
            self._cache[smi] = float(score)
