#!/usr/bin/env python3
"""Greedy or sampled PPO-Bi rollouts with per-episode trajectory logging.

Example: python tools/eval_detailed_trajectories_bi_ppo.py --checkpoint runs/ymhrz9yg/model_step_1001472.pt --run-id ymhrz9yg
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.algorithms.ppo_bi.train import BiPPO, _qed, global_step_from_checkpoint
from src.config import resolve_path

REPO = Path(__file__).resolve().parents[1]

STEP_FIELDS = (
    "path_id",
    "step",
    "reactant",
    "template",
    "template_index",
    "product",
    "qed",
    "reward",
    "second_reactant",
    "action_type",
)

TRAJ_FIELDS = (
    "path_id",
    "start_smiles",
    "final_smiles",
    "initial_qed",
    "final_qed",
    "max_qed",
    "delta_qed",
    "num_reactions",
    "stopped",
)


@dataclass
class StepRow:
    path_id: int
    step: int
    reactant: str
    template: str
    template_index: int | str
    product: str
    qed: float
    reward: float
    second_reactant: str
    action_type: str

    def as_tsv(self) -> str:
        values = [
            self.path_id,
            self.step,
            self.reactant,
            self.template,
            self.template_index,
            self.product,
            f"{self.qed:.6g}",
            f"{self.reward:.6g}",
            self.second_reactant,
            self.action_type,
        ]
        return "\t".join("" if v is None else str(v) for v in values)


def _display_path(path: Path) -> str:
    path = Path(resolve_path(path)).resolve()
    try:
        return str(path.relative_to(REPO.resolve()))
    except ValueError:
        return str(path)


def _trainer_cls(config: dict):
    algo = str(config.get("algorithm", "")).upper()
    if algo == "GRAPHTRANSPPO_BI":
        from src.algorithms.graphtransppo_bi.train import GraphTransBiPPO

        return GraphTransBiPPO
    if algo == "PPO_BI":
        return BiPPO
    raise ValueError(f"Unsupported checkpoint algorithm: {config.get('algorithm')!r}")


def _template_name(trainer: BiPPO, template_index: int) -> str:
    if template_index >= trainer.num_templates:
        return "STOP"
    template = trainer.reaction_manager.templates[template_index]
    return str(template.get("name", template.get("Reaction", template_index)))


def _format_traj_row(item: dict) -> str:
    return "\t".join(str(item.get(k, "")) for k in TRAJ_FIELDS) + "\n"


def _step_reward(trainer: BiPPO, pre_smiles: str, product: str) -> float:
    """Per-step training reward (matches rollout: Δobjective on valid reactions)."""
    if trainer.reward_name == "delta_seh" and trainer.seh_scorer is not None:
        return float(trainer.seh_scorer.step_delta(pre_smiles, product))
    pre_qed = _qed(pre_smiles, round_digits=trainer.qed_round_digits)
    next_qed = _qed(product, round_digits=trainer.qed_round_digits)
    return float(next_qed - pre_qed)


def _record_trajectory(
    path_id: int,
    steps: list[StepRow],
    initial_qed: float,
    *,
    initial_objective: float,
    final_objective: float,
    max_objective: float,
) -> dict:
    final = steps[-1]
    max_qed = max(row.qed for row in steps)
    num_reactions = sum(1 for row in steps[1:] if row.action_type == "REACT")
    stopped = any(row.action_type == "STOP" for row in steps[1:])
    return {
        "path_id": path_id,
        "start_smiles": steps[0].product,
        "final_smiles": final.product,
        "initial_qed": initial_qed,
        "final_qed": final.qed,
        "max_qed": max_qed,
        "delta_qed": final.qed - initial_qed,
        "delta_objective": final_objective - initial_objective,
        "max_objective": max_objective,
        "num_reactions": num_reactions,
        "stopped": int(stopped),
    }


def _trajectory_detailed(
    trainer: BiPPO,
    start_smiles: str,
    path_id: int,
    *,
    deterministic: bool,
) -> tuple[list[StepRow], dict]:
    trainer.policy.eval()
    current = str(start_smiles)
    start_qed = _qed(current, round_digits=trainer.qed_round_digits)
    start_objective = trainer._objective_score(current)
    max_objective = start_objective
    steps: list[StepRow] = [
        StepRow(
            path_id=path_id,
            step=0,
            reactant=current,
            template="START",
            template_index="",
            product=current,
            qed=start_qed,
            reward=0.0,
            second_reactant="",
            action_type="START",
        )
    ]
    react_steps = 0
    with torch.no_grad():
        for _ in range(trainer.max_episode_len + int(trainer.use_stop_action)):
            at_max = react_steps >= trainer.max_episode_len
            if at_max and not trainer.use_stop_action:
                break
            pre_smiles = current
            pre_qed = _qed(pre_smiles, round_digits=trainer.qed_round_digits)
            (
                t_idx,
                r2_idx,
                _log_pi,
                _value,
                _tmpl_mask,
                _r2_mask,
                product,
                _r2_valid_idx,
            ) = trainer._sample_action(
                current, force_stop=at_max, deterministic=deterministic
            )
            step_idx = len(steps)
            post_qed = _qed(current, round_digits=trainer.qed_round_digits)

            if t_idx < 0 or t_idx == trainer.stop_index:
                steps.append(
                    StepRow(
                        path_id=path_id,
                        step=step_idx,
                        reactant=pre_smiles,
                        template="STOP",
                        template_index=trainer.stop_index,
                        product=current,
                        qed=post_qed,
                        reward=0.0,
                        second_reactant="",
                        action_type="STOP",
                    )
                )
                break

            template_name = _template_name(trainer, t_idx)
            r2 = (
                trainer.reactant_keys[r2_idx]
                if 0 <= r2_idx < len(trainer.reactant_keys)
                else ""
            )

            if product is None:
                steps.append(
                    StepRow(
                        path_id=path_id,
                        step=step_idx,
                        reactant=pre_smiles,
                        template=template_name,
                        template_index=t_idx,
                        product=pre_smiles,
                        qed=pre_qed,
                        reward=float(trainer.invalid_reaction_penalty),
                        second_reactant=r2,
                        action_type="INVALID",
                    )
                )
                break

            next_qed = _qed(product, round_digits=trainer.qed_round_digits)
            reward = _step_reward(trainer, pre_smiles, product)
            next_objective = trainer._objective_score(product)
            max_objective = max(max_objective, next_objective)
            steps.append(
                StepRow(
                    path_id=path_id,
                    step=step_idx,
                    reactant=pre_smiles,
                    template=template_name,
                    template_index=t_idx,
                    product=product,
                    qed=next_qed,
                    reward=reward,
                    second_reactant=r2,
                    action_type="REACT",
                )
            )
            current = product
            react_steps += 1

    final_objective = trainer._objective_score(current)
    traj = _record_trajectory(
        path_id,
        steps,
        start_qed,
        initial_objective=start_objective,
        final_objective=final_objective,
        max_objective=max_objective,
    )
    return steps, traj


INFERENCE_MODES = ("greedy", "sampling")


def run_detailed_eval(
    checkpoint: Path,
    *,
    starts_file: Path | None,
    run_id: str | None,
    out_path: Path,
    device: str | None = None,
    max_episodes: int | None = None,
    inference_mode: str = "greedy",
    temperature: float = 1.0,
    start_offset: int = 0,
) -> dict:
    if inference_mode not in INFERENCE_MODES:
        raise ValueError(
            f"inference_mode must be one of {INFERENCE_MODES}, got {inference_mode!r}"
        )
    deterministic = inference_mode == "greedy"
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature!r}")
    checkpoint = Path(resolve_path(checkpoint)).resolve()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = copy.deepcopy(ckpt["config"])
    if starts_file is not None:
        starts_file = Path(resolve_path(starts_file)).resolve()
        config.setdefault("dataset", {})
        config["dataset"]["test_file"] = _display_path(starts_file)

    trainer_cls = _trainer_cls(config)
    trainer = trainer_cls(config)
    trainer.load(checkpoint)
    if device:
        trainer.device = torch.device(device)
        trainer.policy.to(trainer.device)
    # Detailed dumps should sweep the full eval pool, not training n_sampled_eval.
    trainer.sampler.n_sampled_eval = None
    # Greedy (argmax) is invariant to positive scaling, so temperature only
    # has an effect when inference_mode == "sampling".
    trainer.eval_sampling_temperature = temperature

    method_cfg = config.get("graphtransppo_bi") or config.get("ppo_bi") or {}
    algo_label = str(config.get("algorithm", "PPO_BI"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    steps_tmp = out_path.with_suffix(".steps.tmp")
    traj_tmp = out_path.with_suffix(".trajectories.tmp")
    steps_tmp.write_text("", encoding="utf-8")
    traj_tmp.write_text("", encoding="utf-8")

    trainer._swap_active_pool(trainer._eval_pool_role)
    prev_active_keys = trainer._active_r2_keys
    objective_deltas: list[float] = []
    try:
        if not trainer._sparse_r2_graph_encode():
            with torch.no_grad():
                trainer._active_r2_keys = trainer._compute_active_r2_keys(
                    pool=trainer._eval_pool_role, with_grad=False
                )
        else:
            trainer._active_r2_keys = None

        starts = trainer.sampler.eval_starts()
        total_starts = len(starts)
        start_offset = int(start_offset)
        if start_offset:
            starts = starts[start_offset:]
        if max_episodes is not None:
            starts = starts[: int(max_episodes)]
        n_eval = len(starts)

        total_reactions = 0
        best_qed = 0.0
        best_objective = 0.0
        objective_deltas.clear()
        with steps_tmp.open("a", encoding="utf-8") as steps_handle, traj_tmp.open(
            "a", encoding="utf-8"
        ) as traj_handle:
            for local_id, start in enumerate(starts):
                path_id = start_offset + local_id
                steps, traj = _trajectory_detailed(
                    trainer, start, path_id, deterministic=deterministic
                )
                for row in steps:
                    steps_handle.write(row.as_tsv() + "\n")
                traj_handle.write(_format_traj_row(traj))
                total_reactions += int(traj["num_reactions"])
                best_qed = max(best_qed, float(traj["max_qed"]))
                best_objective = max(best_objective, float(traj["max_objective"]))
                objective_deltas.append(float(traj["delta_objective"]))

                if (local_id + 1) % 100 == 0:
                    print(
                        f"  logged {local_id + 1}/{n_eval} episodes "
                        f"(global {start_offset}..{path_id})",
                        flush=True,
                    )
    finally:
        trainer._swap_active_pool("train")
        trainer._active_r2_keys = prev_active_keys

    qed_deltas = []
    with traj_tmp.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            qed_deltas.append(float(parts[6]))
    qed_delta_array = (
        np.asarray(qed_deltas, dtype=np.float64) if qed_deltas else np.array([])
    )
    objective_delta_array = (
        np.asarray(objective_deltas, dtype=np.float64)
        if objective_deltas
        else np.array([])
    )
    reward_name = str(config.get("reward", ""))
    primary_delta_array = (
        objective_delta_array
        if reward_name in {"delta_seh"}
        else qed_delta_array
    )

    starts_display = (
        _display_path(starts_file)
        if starts_file is not None
        else _display_path(Path(resolve_path(config["dataset"]["test_file"])))
    )
    summary = {
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "run_id": run_id or "",
        "checkpoint": _display_path(checkpoint),
        "global_step": global_step_from_checkpoint(checkpoint, ckpt),
        "algorithm": algo_label,
        "inference_mode": inference_mode,
        "sampling_temperature": temperature,
        "reward": config.get("reward", ""),
        "masking": config.get("masking", ""),
        "eval_r2_pool": trainer.eval_r2_pool,
        "r2_arch": trainer.r2_arch,
        "starts_file": starts_display,
        "num_start_molecules": n_eval,
        "start_offset": start_offset,
        "total_starts": total_starts,
        "total_reactions": total_reactions,
        "max_qed": best_qed,
        "best_qed": best_qed,
        "mean_delta_qed": float(qed_delta_array.mean())
        if qed_delta_array.size
        else 0.0,
        "avg_delta_qed": float(qed_delta_array.mean())
        if qed_delta_array.size
        else 0.0,
        "avg_num_reactions": total_reactions / max(n_eval, 1),
        "max_episode_len": int(trainer.max_episode_len),
        "positive_delta_fraction": float((primary_delta_array > 0).mean())
        if primary_delta_array.size
        else 0.0,
        "negative_delta_fraction": float((primary_delta_array < 0).mean())
        if primary_delta_array.size
        else 0.0,
        "zero_delta_fraction": float((primary_delta_array == 0).mean())
        if primary_delta_array.size
        else 0.0,
        "results_file": _display_path(out_path),
    }
    if reward_name == "delta_seh":
        summary["max_seh"] = best_objective
        summary["best_seh"] = best_objective
        summary["mean_delta_seh"] = (
            float(objective_delta_array.mean()) if objective_delta_array.size else 0.0
        )
        summary["avg_delta_seh"] = summary["mean_delta_seh"]

    mode_note = f" ({inference_mode} inference)"
    header = f"# PPO-Bi detailed evaluation results{mode_note}\n"
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(header + "\n")
        handle.write("[summary]\n")
        for key, value in summary.items():
            handle.write(f"{key}: {value}\n")
        handle.write("\n[trajectories]\n")
        handle.write("\t".join(TRAJ_FIELDS) + "\n")
        if traj_tmp.exists():
            handle.write(traj_tmp.read_text(encoding="utf-8"))
        handle.write("\n[steps]\n")
        handle.write("\t".join(STEP_FIELDS) + "\n")
        if steps_tmp.exists():
            handle.write(steps_tmp.read_text(encoding="utf-8"))

    steps_tmp.unlink(missing_ok=True)
    traj_tmp.unlink(missing_ok=True)

    sidecar = out_path.with_suffix(".summary.json")
    sidecar.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--starts-file",
        type=Path,
        default=None,
        help="Override eval start pool (default: config dataset test_file, 2k test set)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: run_detailed_results/ppo or gtppo by algorithm)",
    )
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Output filename prefix (default: {run_id}_detailed)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Limit number of start molecules (per shard count, or smoke test)",
    )
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Skip the first N eval starts (for sharding; combine with --n)",
    )
    parser.add_argument(
        "--inference-mode",
        choices=INFERENCE_MODES,
        default="greedy",
        help="greedy=argmax T and R2; sampling=categorical sample T and R2 each step",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help=(
            "Softmax temperature applied to T and R2 logits before sampling "
            "(only affects --inference-mode sampling). >1 flattens the policy "
            "toward uniform (more diverse), <1 sharpens it. Default 1.0."
        ),
    )
    args = parser.parse_args()

    checkpoint = Path(resolve_path(args.checkpoint))
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    algo = str(ckpt["config"].get("algorithm", "")).upper()
    default_out = REPO / (
        "run_detailed_results/gtppo"
        if algo == "GRAPHTRANSPPO_BI"
        else "run_detailed_results/ppo"
    )
    out_dir = args.out_dir or default_out

    prefix = args.out_prefix or (
        f"{args.run_id}_detailed" if args.run_id else "ppo_detailed"
    )
    out_path = out_dir / f"{prefix}_results.txt"

    print(f"Writing detailed trajectories to {out_path}", flush=True)
    summary = run_detailed_eval(
        checkpoint,
        starts_file=args.starts_file,
        run_id=args.run_id,
        out_path=out_path,
        device=args.device,
        max_episodes=args.n,
        inference_mode=args.inference_mode,
        temperature=args.temperature,
        start_offset=args.start_offset,
    )
    metric_line = (
        f"  episodes={summary['num_start_molecules']} "
        f"mean_delta_seh={summary['mean_delta_seh']:.5f} "
        f"max_seh={summary['max_seh']:.5f}"
        if summary.get("reward") == "delta_seh"
        else f"  episodes={summary['num_start_molecules']} "
        f"mean_delta_qed={summary['mean_delta_qed']:.5f} "
        f"max_qed={summary['max_qed']:.5f}"
    )
    print(metric_line, flush=True)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
