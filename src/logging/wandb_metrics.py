"""Shared W&B metric setup."""

from __future__ import annotations

import wandb


PPO_COMPATIBLE_METRICS = [
    "training/*",
    "train/*",
    "eval/*",
    "training/total_reward_each_episode",
    "train/mean_reward",
    "eval/total_reward_each_episode",
    "eval/mean_reward",
    "eval/mean_ep_length",
    "eval/max_qed",
    "eval/max_episode_qed",
    "eval/mean_final_qed",
    "eval/mean_final_delta_qed",
    "eval/mean_best_delta_qed",
    "eval/mean_final_delta_seh",
    "eval/mean_best_delta_seh",
    "eval/mean_final_seh",
    "eval/max_seh",
    "eval/max_episode_seh",
    "eval/mean_start_seh",
    "eval/avg_delta_qed",
    "eval/std_reward",
    "eval/n_molecules",
    "eval/n_test_molecules",
    "eval/source_train_global_step",
    "eval/episode",
    "eval_count",
    "avg_reward",
    "total_qed",
    "avg_qed",
    "max_qed",
    "episode_length",
    "episode",
    "total_steps",
    "overall_max_qed",
    "total_episodes",
    "cumulative_reward",
    "steps_done",
    "reward_per_step",
    "qed_per_step",
    "seh_per_step",
    "max_seh",
    "overall_max_seh",
    "critic_loss",
    "actor_loss",
    "temperature",
]


def define_ppo_compatible_metrics() -> None:
    if wandb.run is None:
        return
    wandb.define_metric("train/global_step")
    for metric in PPO_COMPATIBLE_METRICS:
        wandb.define_metric(metric, step_metric="train/global_step")
