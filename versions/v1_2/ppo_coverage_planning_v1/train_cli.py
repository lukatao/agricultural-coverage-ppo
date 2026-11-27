"""Standalone CLI entrypoint for training or resuming a PPO agricultural coverage policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import PPOTrainConfig, get_future_scale_presets
from .trainer import train_ppo


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or resume end-to-end PPO agricultural coverage planning.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints, logs, and metrics.")
    parser.add_argument("--experiment-name", type=str, default="ppo_ag_coverage")
    parser.add_argument("--preset", type=str, default=None, choices=list(get_future_scale_presets().keys()))
    parser.add_argument("--resume-from", type=Path, default=None, help="Existing checkpoint bundle directory to resume.")
    parser.add_argument("--dataset-manifest", type=Path, default=None, help="Path to real_fields_splits.json.")
    parser.add_argument("--robot-id", type=str, default="robot_1_reference")
    parser.add_argument("--stage", type=int, default=1, help="Initial curriculum stage (0..7).")
    parser.add_argument("--max-stage", type=int, default=1, help="Maximum curriculum stage (0..7).")
    parser.add_argument("--auto-curriculum", action="store_true", help="Enable validation-driven curriculum progression.")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--dt-action", type=float, default=0.50)
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--subproc", action="store_true")
    parser.add_argument("--reward-preset", type=str, default="simple_v1", choices=["simple_v1", "shaped_v1"])
    parser.add_argument("--local-only", action="store_true", help="Disable global map (local-only ablation).")
    parser.add_argument("--curvature-only", action="store_true", help="Use 1D curvature-only action interface.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()

    if args.device == "cpu" or not torch.cuda.is_available():
        torch.set_num_threads(max(1, args.cpu_threads))

    if args.preset is not None:
        cfg = get_future_scale_presets()[args.preset]
        cfg.experiment_name = args.experiment_name
        cfg.seed = args.seed
        cfg.device = args.device
        if args.dataset_manifest is not None:
            cfg.dataset_manifest = str(args.dataset_manifest)
    else:
        cfg = PPOTrainConfig(
            experiment_name=args.experiment_name,
            robot_id=args.robot_id,
            curriculum_stage=args.stage,
            max_curriculum_stage=max(args.stage, args.max_stage),
            enable_curriculum_progression=args.auto_curriculum,
            dataset_manifest=str(args.dataset_manifest) if args.dataset_manifest else None,
            dt_action_s=args.dt_action,
            include_global_map=(not args.local_only),
            control_interface="curvature_only" if args.curvature_only else "target_speed_curvature",
            reward_preset=args.reward_preset,
            n_envs=args.n_envs,
            use_subproc=args.subproc,
            total_timesteps=args.total_timesteps,
            seed=args.seed,
            device=args.device,
        )

    summary = train_ppo(cfg, args.output_dir, resume_from=args.resume_from, verbose=1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
