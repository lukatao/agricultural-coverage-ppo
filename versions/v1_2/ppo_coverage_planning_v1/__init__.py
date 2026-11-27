"""End-to-End Reinforcement Learning (PPO) for Agricultural Coverage Path Planning."""
from __future__ import annotations

ENV_VERSION = "ppo_coverage_planning_v1.0"

from .config import PPOTrainConfig, get_future_scale_presets
from .coverage_map import CoverageSnapshot, CoverageStepDelta, IncrementalCoverageMap, compute_exact_reference_coverage
from .curriculum import CURRICULUM_STAGES, CurriculumManager, build_validation_suite_for_stage, sample_task_for_stage
from .datasets import AgriculturalFieldDataset, FieldRecord
from .dynamics import ControlInterface, StepDiagnostics, VehicleSimulator, VehicleState
from .env import ObservationConfig, PPOCoverageEnv, PPOEnvConfig
from .field_world import AgriculturalFieldWorld, CollisionMode, CollisionReport
from .networks import AgCoverageFeaturesExtractor
from .rewards import RewardConfig, compute_step_reward, get_reward_preset
from .robot_specs import RobotSpec, get_heterogeneous_robots, get_reference_robot, get_robot_by_id

__all__ = [
    "ENV_VERSION",
    "AgCoverageFeaturesExtractor",
    "AgriculturalFieldDataset",
    "AgriculturalFieldWorld",
    "CURRICULUM_STAGES",
    "CollisionMode",
    "CollisionReport",
    "ControlInterface",
    "CoverageSnapshot",
    "CoverageStepDelta",
    "CurriculumManager",
    "FieldRecord",
    "IncrementalCoverageMap",
    "ObservationConfig",
    "PPOCoverageEnv",
    "PPOEnvConfig",
    "PPOTrainConfig",
    "RewardConfig",
    "RobotSpec",
    "StepDiagnostics",
    "VehicleSimulator",
    "VehicleState",
    "build_validation_suite_for_stage",
    "compute_exact_reference_coverage",
    "compute_step_reward",
    "get_future_scale_presets",
    "get_heterogeneous_robots",
    "get_reference_robot",
    "get_reward_preset",
    "get_robot_by_id",
    "sample_task_for_stage",
]
