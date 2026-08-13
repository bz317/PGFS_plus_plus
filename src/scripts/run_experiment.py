"""PGFS++ training launcher."""

from __future__ import annotations

import argparse
import os

from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a PGFS++ experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-name")
    parser.add_argument("--training-file")
    parser.add_argument("--test-file")
    parser.add_argument("--templates-file")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--run-id")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--n-sampled-eval", type=int)
    parser.add_argument("--eval-freq", type=int)
    parser.add_argument("--save-freq", type=int)
    parser.add_argument("--wandb-resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config["algorithm"] = "PPO_BI"
    if args.training_file or args.test_file or args.templates_file:
        config.setdefault("dataset", {})
        if args.training_file:
            config["dataset"]["training_file"] = args.training_file
        if args.test_file:
            config["dataset"]["test_file"] = args.test_file
        if args.templates_file:
            config["dataset"]["templates_file"] = args.templates_file
    training_cfg = config.setdefault("training", {})
    if args.resume_checkpoint:
        training_cfg["resume_checkpoint"] = args.resume_checkpoint
    if args.run_id:
        training_cfg["run_id"] = args.run_id
    if args.total_timesteps is not None:
        training_cfg["total_timesteps"] = args.total_timesteps
    if args.n_sampled_eval is not None:
        training_cfg["n_sampled_eval"] = args.n_sampled_eval
    if args.eval_freq is not None:
        training_cfg["eval_freq"] = args.eval_freq
    if args.save_freq is not None:
        training_cfg["save_freq"] = args.save_freq
    if args.n_steps is not None:
        config.setdefault("ppo_bi", {})["n_steps"] = args.n_steps
    if args.wandb_resume:
        config["wandb_resume"] = True

    experiment_name = args.experiment_name or config.get("experiment_name", "PGFS++")
    os.environ.setdefault("WANDB_PROJECT", config.get("project", "PGFS++"))

    from src.methods.ppo_bi_adapter import PPOBiAdapter

    PPOBiAdapter.train(config, experiment_name)


if __name__ == "__main__":
    main()
