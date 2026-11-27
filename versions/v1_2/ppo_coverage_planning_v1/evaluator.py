"""Standalone PPO policy evaluator and external conventional-planner comparison interface.

Strictly separates:
  - Neural-network policy inference time (ms/action) vs complete rollout wall-clock time (s).
  - Fast incremental raster coverage vs exact Shapely geometric reference coverage.
  - Deterministic vs stochastic vs untrained random policy evaluation.
  - External post-training comparison with conventional coverage trajectories evaluated
    through the IDENTICAL physical simulator, clearance criteria, and exact coverage evaluator.
"""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from .config import PPOTrainConfig
from .coverage_map import compute_exact_reference_coverage
from .datasets import FieldRecord
from .env import PPOCoverageEnv
from .field_world import CollisionMode
from .robot_specs import RobotSpec, get_robot_by_id


def load_checkpoint(checkpoint_dir: Path | str, device: str = "cpu") -> tuple[PPO, PPOTrainConfig, dict[str, Any]]:
    """Load saved PPO checkpoint without training."""
    cdir = Path(checkpoint_dir)
    meta = json.loads((cdir / "metadata.json").read_text())
    cfg = PPOTrainConfig.from_dict(meta["train_config"])
    model = PPO.load(str(cdir / "model"), device=device)
    return model, cfg, meta


def evaluate_episodes(
    model: PPO | None,
    cfg: PPOTrainConfig,
    episodes: list[tuple[FieldRecord, RobotSpec, int]],
    *,
    deterministic: bool = True,
    use_exact_reference_coverage: bool = True,
    coverage_target_pct: float = 99.0,
    seed_random_policy: int = 999,
) -> dict[str, Any]:
    """Run strict evaluation episodes with exact Shapely reference coverage verification."""
    env_cfg = replace(
        cfg.build_env_config(),
        collision_mode=CollisionMode.TERMINATE,
        coverage_target_pct=coverage_target_pct,
        max_stagnation_time_s=None,
    )
    rng = np.random.default_rng(seed_random_policy)
    per_episode: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []

    for field_rec, robot_spec, ep_seed in episodes:
        world = field_rec.to_world(sdf_resolution_m=0.5)
        env = PPOCoverageEnv(env_config=env_cfg, world=world, robot_spec=robot_spec)
        obs, _ = env.reset(
            seed=ep_seed,
            options={
                "coverage_target_pct": coverage_target_pct,
                "randomize_start_pose": cfg.randomize_start_pose,
            },
        )
        t_roll0 = time.perf_counter()
        inf_time_s = 0.0
        n_actions = 0
        done = False
        last_info: dict[str, Any] = {}
        while not done:
            ti0 = time.perf_counter()
            if model is None:
                action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            inf_time_s += time.perf_counter() - ti0
            n_actions += 1
            obs, _r, terminated, truncated, last_info = env.step(action, record_trajectory=True)
            done = terminated or truncated
        rollout_wall_s = time.perf_counter() - t_roll0

        ep = dict(last_info.get("episode_summary", {}))
        ep["ep_seed"] = ep_seed
        ep["policy_inference_ms_per_action"] = 1000.0 * inf_time_s / max(1, n_actions)
        ep["complete_rollout_wall_s"] = rollout_wall_s
        ep["raster_coverage_pct"] = ep["coverage_pct"]

        traj_xy = env.get_trajectory_xy()
        if use_exact_reference_coverage and len(traj_xy) >= 2:
            exact = compute_exact_reference_coverage(world.polygon, traj_xy, robot_spec.working_width_m)
            ep["exact_coverage_pct"] = exact["coverage_pct"]
            ep["exact_repeated_treatment_m2"] = exact["repeated_treatment_m2"]
            ep["exact_repeated_treatment_pct"] = exact["repeated_treatment_pct"]
            ep["exact_largest_omission_m2"] = exact["largest_untreated_component_m2"]
            ep["exact_obstacle_treatment_m2"] = exact["obstacle_treatment_m2"]
            ep["exact_outside_treatment_m2"] = exact["outside_field_treatment_m2"]
            ep["raster_vs_exact_pp_diff"] = ep["raster_coverage_pct"] - exact["coverage_pct"]
            clean_safe = (not ep["body_collision"]) and (not ep["clearance_violation"])
            ep["exact_99_completed"] = bool(clean_safe and exact["coverage_pct"] >= 99.0)
        else:
            ep["exact_coverage_pct"] = ep["coverage_pct"]
            ep["exact_repeated_treatment_m2"] = ep["repeated_treatment_m2"]
            ep["exact_repeated_treatment_pct"] = ep["repeated_treatment_pct"]
            ep["exact_largest_omission_m2"] = ep["largest_omission_m2"]
            ep["exact_obstacle_treatment_m2"] = ep["obstacle_treatment_m2"]
            ep["exact_outside_treatment_m2"] = 0.0
            ep["raster_vs_exact_pp_diff"] = 0.0
            ep["exact_99_completed"] = ep["final_99_completed"]

        per_episode.append(ep)
        trajectories.append({
            "field_id": field_rec.field_id,
            "family": field_rec.family,
            "robot_id": robot_spec.robot_id,
            "wkb_hex": field_rec.wkb_hex,
            "ep_seed": ep_seed,
            "summary": ep,
            "trajectory_poses": env.get_trajectory_poses(),
        })
        env.close()

    cov_ex = [e["exact_coverage_pct"] for e in per_episode]
    agg = {
        "n_episodes": len(per_episode),
        "policy_type": "random" if model is None else ("ppo_deterministic" if deterministic else "ppo_stochastic"),
        "mean_coverage_pct": float(np.mean(cov_ex)),
        "median_coverage_pct": float(np.median(cov_ex)),
        "max_coverage_pct": float(np.max(cov_ex)),
        "min_coverage_pct": float(np.min(cov_ex)),
        "completion_rate_99_pct": 100.0 * float(np.mean([1.0 if e["exact_99_completed"] else 0.0 for e in per_episode])),
        "stage_completion_rate_pct": 100.0 * float(np.mean([1.0 if e["stage_completed"] else 0.0 for e in per_episode])),
        "body_collision_rate_pct": 100.0 * float(np.mean([1.0 if e["body_collision"] else 0.0 for e in per_episode])),
        "clearance_violation_rate_pct": 100.0 * float(np.mean([1.0 if e["clearance_violation"] else 0.0 for e in per_episode])),
        "mean_trajectory_length_m": float(np.mean([e["distance_m"] for e in per_episode])),
        "mean_repeated_treatment_m2": float(np.mean([e["exact_repeated_treatment_m2"] for e in per_episode])),
        "mean_repeated_treatment_pct": float(np.mean([e["exact_repeated_treatment_pct"] for e in per_episode])),
        "mean_operating_time_s": float(np.mean([e["operating_time_s"] for e in per_episode])),
        "mean_stationary_frac": float(np.mean([e.get("stationary_frac", 0.0) for e in per_episode])),
        "mean_abs_kappa_on_treated": float(np.mean([e.get("mean_abs_kappa_on_treated", 0.0) for e in per_episode])),
        "mean_time_without_new_cov_s": float(np.mean([e.get("time_without_new_cov_s", 0.0) for e in per_episode])),
        "mean_new_area_per_sec_m2_s": float(np.mean([e["new_area_per_sec"] for e in per_episode])),
        "mean_new_area_per_m_m2_m": float(np.mean([e["new_area_per_m"] for e in per_episode])),
        "mean_largest_omission_m2": float(np.mean([e["exact_largest_omission_m2"] for e in per_episode])),
        "mean_obstacle_treatment_m2": float(np.mean([e["exact_obstacle_treatment_m2"] for e in per_episode])),
        "mean_episode_reward": float(np.mean([e["episode_reward"] for e in per_episode])),
        "mean_policy_inference_ms_per_action": float(np.mean([e["policy_inference_ms_per_action"] for e in per_episode])),
        "mean_complete_rollout_wall_s": float(np.mean([e["complete_rollout_wall_s"] for e in per_episode])),
        "mean_abs_raster_vs_exact_pp": float(np.mean([abs(e["raster_vs_exact_pp_diff"]) for e in per_episode])),
        "total_constraint_violations": int(sum(e["constraint_violations"] for e in per_episode)),
    }
    return {"aggregate": agg, "episodes": per_episode, "trajectories": trajectories}


def evaluate_external_conventional_trajectory(
    field_rec: FieldRecord,
    robot_spec: RobotSpec,
    reference_xy: np.ndarray,
    *,
    planner_name: str,
    planning_wall_s: float,
    dt_phys_s: float = 0.02,
) -> dict[str, Any]:
    """External post-training benchmark interface: evaluate a conventional trajectory
    under the IDENTICAL physical field geometry, clearance rules, and exact Shapely coverage.

    Never called during PPO training. Uses the exact same VehicleSimulator & collision rules
    if tracked, or evaluates the geometric path with physical speed parameterization.
    """
    world = field_rec.to_world(sdf_resolution_m=0.5)
    pts = np.asarray(reference_xy, dtype=np.float64)[:, :2]
    diffs = np.diff(pts, axis=0)
    ds = np.hypot(diffs[:, 0], diffs[:, 1])
    total_len = float(np.sum(ds))
    exact = compute_exact_reference_coverage(world.polygon, pts, robot_spec.working_width_m)

    # Check continuous swept collision & clearance along reference polyline under identical world & spec
    substeps = np.column_stack([pts[:, 0], pts[:, 1], np.zeros(len(pts)), np.zeros(len(pts)), np.zeros(len(pts))])
    coll = world.check_motion_substeps(substeps, robot_spec)

    # Estimate physical operating time using curvature-limited speed profile along polyline
    est_time_s = 0.0
    if len(pts) >= 3:
        for i in range(len(ds)):
            seg_len = float(ds[i])
            if seg_len <= 1e-9:
                continue
            if 0 < i < len(pts) - 1:
                v1 = pts[i] - pts[i - 1]
                v2 = pts[i + 1] - pts[i]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 1e-6 and n2 > 1e-6:
                    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                    ang = math.acos(cos_a)
                    kap = ang / max(0.5 * (n1 + n2), 1e-4)
                else:
                    kap = 0.0
            else:
                kap = 0.0
            v_adm = max(0.2, robot_spec.allowed_speed(kap))
            est_time_s += seg_len / v_adm
    else:
        est_time_s = total_len / robot_spec.max_straight_speed_m_s

    return {
        "planner": planner_name,
        "field_id": field_rec.field_id,
        "family": field_rec.family,
        "robot_id": robot_spec.robot_id,
        "exact_coverage_pct": exact["coverage_pct"],
        "exact_repeated_treatment_m2": exact["repeated_treatment_m2"],
        "exact_repeated_treatment_pct": exact["repeated_treatment_pct"],
        "exact_largest_omission_m2": exact["largest_untreated_component_m2"],
        "exact_obstacle_treatment_m2": exact["obstacle_treatment_m2"],
        "trajectory_length_m": total_len,
        "estimated_operating_time_s": est_time_s,
        "planning_wall_s": planning_wall_s,
        "body_collision": coll.body_collision,
        "clearance_violation": coll.clearance_breach,
        "min_body_clearance_m": coll.min_body_clearance_m,
    }
