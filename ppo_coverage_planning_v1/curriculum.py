"""Configurable 8-stage autonomous RL curriculum (Stages 0-7) and validation-driven progression.

Curriculum changes environmental difficulty ONLY — never provides expert trajectories,
swaths, BCD cells, or conventional solutions.

Progression is governed strictly by validation agricultural & safety metrics (median
coverage, completion rate, collision rate, safety-clearance violation rate) rather than
merely elapsed timesteps, while preserving previous-stage retention tracking.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .datasets import (
    AgriculturalFieldDataset,
    FieldRecord,
    generate_open_region,
    generate_rectangle_field,
    generate_synthetic_convex,
    generate_synthetic_nonconvex,
    generate_synthetic_with_obstacles,
)
from .robot_specs import (
    RobotSpec,
    get_heterogeneous_robots,
    get_reference_robot,
    sample_synthetic_robot_spec,
)


@dataclass(frozen=True)
class ProgressionCriteria:
    """Registered validation thresholds required before advancing to the next curriculum stage."""
    min_median_coverage_pct: float
    min_completion_rate: float
    max_body_collision_rate: float
    max_clearance_violation_rate: float
    min_consecutive_passes: int = 2


@dataclass(frozen=True)
class CurriculumStageConfig:
    stage_id: int
    name: str
    description: str
    coverage_target_pct: float        # Intermediate training target (always separately track unconditional 99%!)
    time_budget_multiplier: float     # Episode max_sim_time_s = max(min_s, multiplier * T_scale)
    min_episode_time_s: float
    randomize_start_pose: bool
    randomize_robot: bool
    allowed_families: tuple[str, ...]
    progression: ProgressionCriteria


CURRICULUM_STAGES: dict[int, CurriculumStageConfig] = {
    0: CurriculumStageConfig(
        stage_id=0,
        name="stage0_open_sanity",
        description="Large open 100 m x 100 m region sanity check: verify PPO learns productive forward coverage.",
        coverage_target_pct=12.0,
        time_budget_multiplier=0.14,
        min_episode_time_s=120.0,
        randomize_start_pose=True,
        randomize_robot=False,
        allowed_families=("A",),
        progression=ProgressionCriteria(10.0, 0.50, 0.05, 0.05, 1),
    ),
    1: CurriculumStageConfig(
        stage_id=1,
        name="stage1_small_rectangles",
        description="Small rectangular fields (24-30 m x 20-26 m) with legal turning room and randomized start pose.",
        coverage_target_pct=90.0,
        time_budget_multiplier=3.5,
        min_episode_time_s=150.0,
        randomize_start_pose=True,
        randomize_robot=False,
        allowed_families=("A",),
        progression=ProgressionCriteria(85.0, 0.60, 0.15, 0.20, 2),
    ),
    2: CurriculumStageConfig(
        stage_id=2,
        name="stage2_diverse_rectangles",
        description="Randomized rectangle dimensions (22-46 m), aspect ratios (0.5-2.0), orientations (0-180 deg), and starts.",
        coverage_target_pct=95.0,
        time_budget_multiplier=3.5,
        min_episode_time_s=180.0,
        randomize_start_pose=True,
        randomize_robot=False,
        allowed_families=("A",),
        progression=ProgressionCriteria(90.0, 0.65, 0.12, 0.15, 2),
    ),
    3: CurriculumStageConfig(
        stage_id=3,
        name="stage3_convex_fields",
        description="Synthetic convex polygons + small/medium real EU Family A agricultural fields.",
        coverage_target_pct=97.0,
        time_budget_multiplier=3.5,
        min_episode_time_s=200.0,
        randomize_start_pose=True,
        randomize_robot=False,
        allowed_families=("A",),
        progression=ProgressionCriteria(93.0, 0.70, 0.10, 0.15, 2),
    ),
    4: CurriculumStageConfig(
        stage_id=4,
        name="stage4_nonconvex_fields",
        description="Family B non-convex fields (synthetic L/U shapes + real EU Family B fields).",
        coverage_target_pct=98.0,
        time_budget_multiplier=4.0,
        min_episode_time_s=240.0,
        randomize_start_pose=True,
        randomize_robot=False,
        allowed_families=("A", "B"),
        progression=ProgressionCriteria(94.0, 0.70, 0.10, 0.15, 2),
    ),
    5: CurriculumStageConfig(
        stage_id=5,
        name="stage5_obstacle_fields",
        description="Family C convex fields with interior obstacles (synthetic + real EU Family C fields).",
        coverage_target_pct=99.0,
        time_budget_multiplier=4.2,
        min_episode_time_s=260.0,
        randomize_start_pose=True,
        randomize_robot=False,
        allowed_families=("A", "B", "C"),
        progression=ProgressionCriteria(95.0, 0.75, 0.08, 0.12, 2),
    ),
    6: CurriculumStageConfig(
        stage_id=6,
        name="stage6_complex_fields",
        description="Family D complex non-convex fields with multiple obstacles and corridors (synthetic + real EU Family D).",
        coverage_target_pct=99.0,
        time_budget_multiplier=4.5,
        min_episode_time_s=300.0,
        randomize_start_pose=True,
        randomize_robot=False,
        allowed_families=("A", "B", "C", "D"),
        progression=ProgressionCriteria(96.0, 0.75, 0.05, 0.10, 2),
    ),
    7: CurriculumStageConfig(
        stage_id=7,
        name="stage7_heterogeneous_robots",
        description="All field families (A-D) with randomized heterogeneous RobotSpec (4 registered/synthetic + sampled).",
        coverage_target_pct=99.0,
        time_budget_multiplier=4.5,
        min_episode_time_s=300.0,
        randomize_start_pose=True,
        randomize_robot=True,
        allowed_families=("A", "B", "C", "D"),
        progression=ProgressionCriteria(96.0, 0.75, 0.05, 0.10, 2),
    ),
}


def sample_task_for_stage(
    stage_id: int,
    rng: np.random.Generator,
    *,
    dataset: AgriculturalFieldDataset | None = None,
    split: str = "train",
    fixed_robot: RobotSpec | None = None,
    allow_sampled_robots: bool = False,
) -> tuple[FieldRecord, RobotSpec]:
    """Deterministically sample a compatible (FieldRecord, RobotSpec) pair for a curriculum stage."""
    if stage_id not in CURRICULUM_STAGES:
        raise ValueError(f"Invalid curriculum stage {stage_id}; expected 0..7")
    cfg = CURRICULUM_STAGES[stage_id]

    # Select RobotSpec
    if fixed_robot is not None:
        robot = fixed_robot
    elif cfg.randomize_robot:
        robots = list(get_heterogeneous_robots().values())
        if allow_sampled_robots and rng.random() < 0.25:
            robot = sample_synthetic_robot_spec(rng)
        else:
            robot = robots[int(rng.integers(0, len(robots)))]
    else:
        robot = get_reference_robot()

    # Select or generate FieldRecord compatible with robot
    idx_tag = int(rng.integers(0, 1_000_000))
    if stage_id == 0:
        rec = generate_open_region(size_m=100.0, field_id=f"s0_open_{idx_tag}")
    elif stage_id == 1:
        w = float(rng.uniform(24.0, 30.0))
        h = float(rng.uniform(20.0, 26.0))
        rec = generate_rectangle_field(w, h, angle_deg=0.0, field_id=f"s1_rect_{idx_tag}")
    elif stage_id == 2:
        w = float(rng.uniform(22.0, 44.0))
        h = float(rng.uniform(20.0, 40.0))
        ang = float(rng.uniform(0.0, 180.0))
        rec = generate_rectangle_field(w, h, angle_deg=ang, field_id=f"s2_rect_{idx_tag}")
    elif stage_id == 3:
        real_a = dataset.get_records(families=("A",), split=split, max_area_m2=8000.0) if dataset else []
        if real_a and rng.random() < 0.55:
            rec = real_a[int(rng.integers(0, len(real_a)))]
        else:
            rec = generate_synthetic_convex(rng, radius_range_m=(16.0, 28.0), field_id=f"s3_conv_{idx_tag}")
    elif stage_id == 4:
        real_b = dataset.get_records(families=("B",), split=split, max_area_m2=15000.0) if dataset else []
        if real_b and rng.random() < 0.45:
            rec = real_b[int(rng.integers(0, len(real_b)))]
        else:
            rec = generate_synthetic_nonconvex(rng, scale_range_m=(34.0, 52.0), field_id=f"s4_nonconv_{idx_tag}")
    elif stage_id == 5:
        real_c = dataset.get_records(families=("C",), split=split, max_area_m2=18000.0) if dataset else []
        if real_c and rng.random() < 0.45:
            rec = real_c[int(rng.integers(0, len(real_c)))]
        else:
            rec = generate_synthetic_with_obstacles(
                rng, nonconvex_exterior=False, n_obstacles=1, scale_range_m=(42.0, 58.0), field_id=f"s5_obs_{idx_tag}"
            )
    else:
        # Stage 6 or Stage 7
        real_all = (
            dataset.get_records(families=cfg.allowed_families, split=split, max_area_m2=25000.0)
            if dataset
            else []
        )
        if real_all and rng.random() < 0.50:
            rec = real_all[int(rng.integers(0, len(real_all)))]
        else:
            u = rng.random()
            if u < 0.25:
                rec = generate_synthetic_convex(rng, field_id=f"s{stage_id}_conv_{idx_tag}")
            elif u < 0.50:
                rec = generate_synthetic_nonconvex(rng, field_id=f"s{stage_id}_nonconv_{idx_tag}")
            elif u < 0.75:
                rec = generate_synthetic_with_obstacles(rng, nonconvex_exterior=False, n_obstacles=1, field_id=f"s{stage_id}_c_{idx_tag}")
            else:
                rec = generate_synthetic_with_obstacles(rng, nonconvex_exterior=True, n_obstacles=2, field_id=f"s{stage_id}_d_{idx_tag}")

    # Guarantee robot navigable compatibility (fallback to suitably sized rectangle for heavy wide robots)
    world = rec.to_world(sdf_resolution_m=1.0)
    if world.navigable_domain(robot, extra_margin_m=0.25).is_empty:
        scale = max(36.0, 8.0 * robot.min_turning_radius_m)
        rec = generate_rectangle_field(scale, 0.85 * scale, angle_deg=float(rng.uniform(0, 180)), field_id=f"s{stage_id}_compat_{idx_tag}")
    return rec, robot


def build_validation_suite_for_stage(
    stage_id: int,
    *,
    dataset: AgriculturalFieldDataset | None = None,
    n_episodes: int = 8,
    seed: int = 1000,
    fixed_robot: RobotSpec | None = None,
) -> list[tuple[FieldRecord, RobotSpec, int]]:
    """Build a deterministic fixed validation episode list (field, robot, reset_seed) for a stage."""
    rng = np.random.default_rng(seed + 97 * stage_id)
    suite: list[tuple[FieldRecord, RobotSpec, int]] = []
    for k in range(n_episodes):
        rec, robot = sample_task_for_stage(
            stage_id, rng, dataset=dataset, split="val", fixed_robot=fixed_robot
        )
        suite.append((rec, robot, seed + 1000 * stage_id + k))
    return suite


class CurriculumManager:
    """Validation-driven curriculum progression state machine with serializable checkpoints."""

    def __init__(self, initial_stage: int = 1, max_stage: int = 7) -> None:
        if initial_stage not in CURRICULUM_STAGES or max_stage not in CURRICULUM_STAGES:
            raise ValueError("Stages must be in 0..7")
        self.current_stage = int(initial_stage)
        self.max_stage = int(max_stage)
        self.consecutive_passes = 0
        self.history: list[dict[str, Any]] = []

    @property
    def stage_config(self) -> CurriculumStageConfig:
        return CURRICULUM_STAGES[self.current_stage]

    def evaluate_progression(self, val_metrics: dict[str, float], timesteps: int) -> tuple[bool, str]:
        """Decide whether to advance stage based purely on validation agricultural & safety metrics."""
        crit = self.stage_config.progression
        med_cov = float(val_metrics.get("median_coverage_pct", 0.0))
        comp_rate = float(val_metrics.get("stage_completion_rate", 0.0))
        body_coll = float(val_metrics.get("body_collision_rate", 1.0))
        clr_viol = float(val_metrics.get("clearance_violation_rate", 1.0))

        passed = (
            med_cov >= crit.min_median_coverage_pct
            and comp_rate >= crit.min_completion_rate
            and body_coll <= crit.max_body_collision_rate
            and clr_viol <= crit.max_clearance_violation_rate
        )
        if passed:
            self.consecutive_passes += 1
        else:
            self.consecutive_passes = 0

        advanced = False
        reason = "criteria_not_met"
        if passed and self.consecutive_passes >= crit.min_consecutive_passes:
            if self.current_stage < self.max_stage:
                self.current_stage += 1
                self.consecutive_passes = 0
                advanced = True
                reason = f"advanced_to_stage_{self.current_stage}"
            else:
                reason = "max_stage_passed"
        elif passed:
            reason = f"passed_{self.consecutive_passes}_of_{crit.min_consecutive_passes}"

        self.history.append({
            "timesteps": int(timesteps),
            "stage_evaluated": self.current_stage - 1 if advanced else self.current_stage,
            "advanced": advanced,
            "new_stage": self.current_stage,
            "reason": reason,
            "median_coverage_pct": med_cov,
            "stage_completion_rate": comp_rate,
            "body_collision_rate": body_coll,
            "clearance_violation_rate": clr_viol,
        })
        return advanced, reason

    def state_dict(self) -> dict[str, Any]:
        return {
            "current_stage": self.current_stage,
            "max_stage": self.max_stage,
            "consecutive_passes": self.consecutive_passes,
            "history": list(self.history),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.current_stage = int(state["current_stage"])
        self.max_stage = int(state["max_stage"])
        self.consecutive_passes = int(state.get("consecutive_passes", 0))
        self.history = list(state.get("history", []))
