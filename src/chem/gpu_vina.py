"""QuickVina2-GPU docking for optional Vina-based reward scoring."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from src.config import project_root

DEFAULT_VINA_DIR = project_root() / "bin" / "QuickVina2-GPU-2-1"
DEFAULT_VINA_EXE = DEFAULT_VINA_DIR / "Vina-GPU"
DEFAULT_OPENCL_BINARY = DEFAULT_VINA_DIR
DOCKING_DATA_DIR = project_root() / "data" / "docking"
TARGETS_MANIFEST = DOCKING_DATA_DIR / "targets.json"


def _load_targets() -> dict:
    if TARGETS_MANIFEST.is_file():
        with TARGETS_MANIFEST.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            out = {}
            for key, info in payload.items():
                entry = dict(info)
                receptor = entry.get("receptor")
                if receptor and not Path(receptor).is_absolute():
                    entry["receptor"] = str(project_root() / receptor)
                out[str(key).lower()] = entry
            return out
    # Fallback if manifest is missing (kras only).
    return {
        "kras": {
            "receptor": str(DOCKING_DATA_DIR / "kras" / "8azr.pdbqt"),
            "center_x": 21.466,
            "center_y": -0.650,
            "center_z": 5.028,
            "size_x": 18,
            "size_y": 18,
            "size_z": 18,
            "num_atoms": 32,
        }
    }


TARGETS = _load_targets()


def gpu_vina_installed(vina_path: str | Path | None = None) -> bool:
    path = Path(vina_path or DEFAULT_VINA_EXE)
    return path.is_file() and os.access(path, os.X_OK)


def _find_obabel() -> str | None:
    exe = shutil.which("obabel")
    if exe:
        return exe
    candidate = Path(os.environ.get("CONDA_PREFIX", "")) / "bin" / "obabel"
    if candidate.is_file():
        return str(candidate)
    return None


def smiles_to_pdbqt(smiles: str, pdbqt_file: str | Path, *, obabel: str | None = None) -> bool:
    """Prepare a ligand PDBQT from SMILES using Open Babel."""
    obabel = obabel or _find_obabel()
    if obabel is None:
        return False
    result = subprocess.run(
        [obabel, f"-:{smiles}", "-opdbqt", "--gen3d", "-O", str(pdbqt_file)],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and Path(pdbqt_file).is_file()


def parse_affinity_from_pdbqt(pdbqt_file: str | Path) -> float | None:
    with Path(pdbqt_file).open("r", encoding="utf-8") as fh:
        for line in fh:
            if "REMARK VINA RESULT" in line:
                return float(line.split()[3])
    return None


class QuickVina2GPU:
    """Batch QuickVina2-GPU scorer for a single fixed target."""

    def __init__(
        self,
        *,
        vina_path: str | Path | None = None,
        opencl_binary_path: str | Path | None = None,
        target: str | None = None,
        save_confs: bool = False,
        reward_scale_max: float = -1.0,
        reward_scale_min: float = -10.0,
        thread: int = 8000,
        print_time: bool = False,
        print_logs: bool = False,
        obabel_path: str | None = None,
    ) -> None:
        self.vina_path = str(vina_path or DEFAULT_VINA_EXE)
        self.opencl_binary_path = str(opencl_binary_path or DEFAULT_OPENCL_BINARY)
        self.save_confs = bool(save_confs)
        self.thread = int(thread)
        self.print_time = bool(print_time)
        self.print_logs = bool(print_logs)
        self.reward_scale_max = float(reward_scale_max)
        self.reward_scale_min = float(reward_scale_min)
        self.obabel_path = obabel_path or _find_obabel()

        if target is None:
            raise ValueError("QuickVina2GPU requires `target`")
        key = str(target).lower()
        if key not in TARGETS:
            known = ", ".join(sorted(TARGETS))
            raise ValueError(f"Unknown Vina target {target!r} (known: {known})")
        self.target_key = key
        self.target_info = dict(TARGETS[key])
        for attr, value in self.target_info.items():
            setattr(self, attr, value)
        receptor = Path(self.receptor)
        if not receptor.is_file():
            raise FileNotFoundError(
                f"Receptor PDBQT not found for target {target!r}: {receptor}. "
                f"Run scripts/prepare_vina_docking_data.py."
            )
        if not gpu_vina_installed(self.vina_path):
            raise FileNotFoundError(
                f"QuickVina2-GPU executable not found or not executable: {self.vina_path}. "
                "Build Vina-GPU-2.1 and place it under bin/QuickVina2-GPU-2-1/."
            )
        if self.obabel_path is None:
            raise FileNotFoundError(
                "Open Babel (`obabel`) is required for ligand preparation. "
                "Install via conda-forge: conda install -c conda-forge openbabel"
            )

    def _write_config_file(self, config_path: Path, input_dir: Path) -> None:
        lines = [
            f"receptor = {self.receptor}",
            f"ligand_directory = {input_dir}",
            f"opencl_binary_path = {self.opencl_binary_path}",
            f"center_x = {self.center_x}",
            f"center_y = {self.center_y}",
            f"center_z = {self.center_z}",
            f"size_x = {self.size_x}",
            f"size_y = {self.size_y}",
            f"size_z = {self.size_z}",
            f"thread = {self.thread}",
        ]
        config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_pdbqt_files(self, smiles: Sequence[str], input_dir: Path) -> None:
        for idx, smi in enumerate(smiles):
            out = input_dir / f"input_{idx}.pdbqt"
            if not smiles_to_pdbqt(smi, out, obabel=self.obabel_path):
                continue

    def _run_vina(self, config_path: Path) -> bool:
        result = subprocess.run(
            [self.vina_path, "--config", str(config_path)],
            capture_output=True,
            text=True,
        )
        if self.print_time and result.stdout:
            lines = result.stdout.splitlines()
            if lines:
                print(lines[-1])
        if self.print_logs and result.stdout:
            print(result.stdout)
        if result.returncode != 0:
            print(f"Vina failed with return code {result.returncode}")
            if result.stderr:
                print(result.stderr)
            return False
        return True

    def _parse_results(self, batch_size: int, out_dir: Path) -> list[float]:
        affinities: list[float] = []
        failed = 0
        for idx in range(batch_size):
            pdbqt_file = out_dir / f"input_{idx}_out.pdbqt"
            if pdbqt_file.is_file():
                affinity = parse_affinity_from_pdbqt(pdbqt_file)
                affinities.append(float(affinity) if affinity is not None else 0.0)
            else:
                affinities.append(0.0)
                failed += 1
        if failed:
            print(f"WARNING: Failed to calculate affinity for {failed}/{batch_size} molecules")
        return affinities

    def _affinity_to_reward(self, affinities: Iterable[float]) -> list[float]:
        arr = np.asarray(list(affinities), dtype=float)
        rewards = (arr + self.reward_scale_min) / (self.reward_scale_min + self.reward_scale_max) - 1.0
        return list(rewards.astype(float))

    def calculate_rewards(self, smiles: Sequence[str]) -> tuple[list[str], list[float], list[float]]:
        batch_size = len(smiles)
        if batch_size == 0:
            return [], [], []

        with tempfile.TemporaryDirectory(prefix="src_vina_") as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "in"
            out_dir = tmp_path / "in_out"
            config_path = tmp_path / "config.txt"
            input_dir.mkdir()
            out_dir.mkdir()

            self._write_pdbqt_files(smiles, input_dir)
            self._write_config_file(config_path, input_dir)
            ok = self._run_vina(config_path)
            if not ok or not out_dir.is_dir():
                affinities = [0.0] * batch_size
            else:
                affinities = self._parse_results(batch_size, out_dir)

            rewards = self._affinity_to_reward(affinities)
            num_atoms_limit = int(self.target_info.get("num_atoms", 0)) + 8
            for idx, smi in enumerate(smiles):
                if num_atoms_limit <= 0:
                    continue
                mol = None
                try:
                    from rdkit import Chem

                    mol = Chem.MolFromSmiles(smi)
                except Exception:
                    mol = None
                if mol is not None and mol.GetNumHeavyAtoms() > num_atoms_limit:
                    rewards[idx] -= 0.4

            if affinities:
                print(
                    "AFFINITIES:",
                    f"mean={round(float(np.mean(affinities)), 3)},",
                    f"std={round(float(np.std(affinities)), 3)},",
                    f"min={round(float(np.min(affinities)), 3)},",
                    f"max={round(float(np.max(affinities)), 3)}",
                )
            return list(smiles), affinities, rewards
