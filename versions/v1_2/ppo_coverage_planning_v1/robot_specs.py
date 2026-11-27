"""Portable heterogeneous agricultural robot specifications for PPO coverage planning.

Preserves the exact registered reference robot parameters from the thesis repository
(coverage_benchmark/contracts.py, field_map_manager/config/field.yaml,
bunker_nav2/config/limits.yaml) without requiring local ROS2 or coverage_benchmark
imports so the training package runs cleanly on Kaggle and local machines alike.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np


# Documented normalization scales for the 12-D robot capability observation vector.
CAPABILITY_SCALES: dict[str, float] = {
    "working_width_m": 10.0,
    "max_straight_speed_m_s": 3.0,
    "max_turning_speed_m_s": 1.5,
    "min_turning_radius_m": 10.0,
    "max_curvature_inv_m": 1.0,
    "max_curvature_rate_inv_m2": 0.5,
    "max_linear_accel_m_s2": 1.5,
    "max_linear_decel_m_s2": 1.5,
    "body_radius_m": 2.0,
    "exterior_center_clearance_m": 3.0,
    "has_ackermann": 1.0,
    "max_steering_angle_rad": math.pi / 2.0,
}

CAPABILITY_FEATURE_NAMES: tuple[str, ...] = tuple(CAPABILITY_SCALES.keys())


@dataclass(frozen=True)
class EnergyModelSpec:
    """Optional parametric energy model (disabled for reference robot where hardware specs are absent)."""
    enabled: bool = False
    status: str = "unverified_unavailable"
    battery_capacity_wh: float | None = None
    initial_soc: float = 1.0
    min_operating_soc: float = 0.0
    auxiliary_power_w: float = 0.0
    tool_power_w: float = 0.0
    distance_energy_j_per_m: float = 0.0
    speed_cubic_coeff_w_s3_m3: float = 0.0
    turning_energy_j_per_rad: float = 0.0

    def power_watts(self, speed_m_s: float, curvature_inv_m: float, tool_active: bool) -> float:
        if not self.enabled:
            return 0.0
        v = max(0.0, float(speed_m_s))
        omega = abs(v * float(curvature_inv_m))
        return (
            self.auxiliary_power_w
            + (self.tool_power_w if tool_active else 0.0)
            + self.distance_energy_j_per_m * v
            + self.speed_cubic_coeff_w_s3_m3 * (v ** 3)
            + self.turning_energy_j_per_rad * omega
        )


@dataclass(frozen=True)
class RobotSpec:
    """Complete configurable physical specification for a single agricultural robot."""
    robot_id: str
    display_name: str
    provenance: str  # "registered_reference", "synthetic_experimental", or "sampled_synthetic"

    body_radius_m: float
    boundary_clearance_m: float
    obstacle_clearance_m: float
    working_width_m: float

    max_straight_speed_m_s: float
    max_turning_speed_m_s: float
    min_turning_radius_m: float
    max_curvature_rate_inv_m2: float
    max_angular_velocity_rad_s: float
    max_linear_accel_m_s2: float
    max_linear_decel_m_s2: float
    max_angular_accel_rad_s2: float

    forward_only: bool = True
    allow_in_place_rotation: bool = False
    tool_always_active: bool = True

    wheelbase_m: float | None = None
    max_steering_angle_rad: float | None = None
    max_steering_rate_rad_s: float | None = None

    implement_collision_status: str = "unverified"
    energy: EnergyModelSpec = field(default_factory=EnergyModelSpec)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.body_radius_m <= 0.0 or self.working_width_m <= 0.0:
            raise ValueError("body_radius_m and working_width_m must be positive")
        if self.min_turning_radius_m <= 0.0 or self.max_curvature_rate_inv_m2 <= 0.0:
            raise ValueError("min_turning_radius_m and max_curvature_rate_inv_m2 must be positive")
        if not (0.0 < self.max_turning_speed_m_s <= self.max_straight_speed_m_s):
            raise ValueError("Expected 0 < max_turning_speed_m_s <= max_straight_speed_m_s")
        if self.max_linear_accel_m_s2 <= 0.0 or self.max_linear_decel_m_s2 <= 0.0:
            raise ValueError("Acceleration and deceleration limits must be positive")
        if self.max_angular_velocity_rad_s <= 0.0 or self.max_angular_accel_rad_s2 <= 0.0:
            raise ValueError("Angular velocity and angular acceleration limits must be positive")
        if self.wheelbase_m is not None:
            if self.wheelbase_m <= 0.0:
                raise ValueError("wheelbase_m must be positive")
            if self.max_steering_angle_rad is not None:
                k_steer = math.tan(self.max_steering_angle_rad) / self.wheelbase_m
                if k_steer + 1e-9 < (1.0 / self.min_turning_radius_m):
                    raise ValueError(
                        f"Ackermann limit tan(delta_max)/L={k_steer:.4f} tighter than 1/R_min={1.0/self.min_turning_radius_m:.4f}"
                    )

    @property
    def max_curvature_inv_m(self) -> float:
        k_r = 1.0 / self.min_turning_radius_m
        if self.wheelbase_m is not None and self.max_steering_angle_rad is not None:
            return min(k_r, math.tan(self.max_steering_angle_rad) / self.wheelbase_m)
        return k_r

    @property
    def implied_max_lateral_accel_m_s2(self) -> float:
        return (self.max_turning_speed_m_s ** 2) / self.min_turning_radius_m

    @property
    def exterior_center_clearance_m(self) -> float:
        return self.body_radius_m + self.boundary_clearance_m

    @property
    def obstacle_center_clearance_m(self) -> float:
        return self.body_radius_m + self.obstacle_clearance_m

    def allowed_speed(self, curvature_inv_m: float) -> float:
        ak = abs(float(curvature_inv_m))
        if ak <= 1e-12:
            return self.max_straight_speed_m_s
        v_omega = self.max_angular_velocity_rad_s / ak
        v_lat = math.sqrt(self.implied_max_lateral_accel_m_s2 / ak)
        return min(self.max_straight_speed_m_s, v_omega, v_lat)

    def max_safe_curvature(self, speed_m_s: float) -> float:
        v = max(0.0, float(speed_m_s))
        kmax = self.max_curvature_inv_m
        if v <= 1e-9:
            return kmax
        k_omega = self.max_angular_velocity_rad_s / v
        k_lat = self.implied_max_lateral_accel_m_s2 / (v * v)
        return min(kmax, k_omega, k_lat)

    def max_temporal_curvature_rate(self, speed_m_s: float, curvature_inv_m: float) -> float:
        v = max(0.0, float(speed_m_s))
        if v <= 1e-12 and not self.allow_in_place_rotation:
            return 0.0
        dk_spatial = v * self.max_curvature_rate_inv_m2
        if self.wheelbase_m is not None and self.max_steering_rate_rad_s is not None:
            L = self.wheelbase_m
            dk_steer = ((1.0 + (L * curvature_inv_m) ** 2) / L) * self.max_steering_rate_rad_s
            return min(dk_spatial, dk_steer)
        return dk_spatial

    def to_capability_vector(self) -> np.ndarray:
        """Return 12-D normalized capability vector using documented CAPABILITY_SCALES."""
        has_ack = 1.0 if self.wheelbase_m is not None else 0.0
        steer_norm = (
            float(self.max_steering_angle_rad) / CAPABILITY_SCALES["max_steering_angle_rad"]
            if (self.wheelbase_m is not None and self.max_steering_angle_rad is not None)
            else 0.0
        )
        vec = np.array([
            self.working_width_m / CAPABILITY_SCALES["working_width_m"],
            self.max_straight_speed_m_s / CAPABILITY_SCALES["max_straight_speed_m_s"],
            self.max_turning_speed_m_s / CAPABILITY_SCALES["max_turning_speed_m_s"],
            self.min_turning_radius_m / CAPABILITY_SCALES["min_turning_radius_m"],
            self.max_curvature_inv_m / CAPABILITY_SCALES["max_curvature_inv_m"],
            self.max_curvature_rate_inv_m2 / CAPABILITY_SCALES["max_curvature_rate_inv_m2"],
            self.max_linear_accel_m_s2 / CAPABILITY_SCALES["max_linear_accel_m_s2"],
            self.max_linear_decel_m_s2 / CAPABILITY_SCALES["max_linear_decel_m_s2"],
            self.body_radius_m / CAPABILITY_SCALES["body_radius_m"],
            self.exterior_center_clearance_m / CAPABILITY_SCALES["exterior_center_clearance_m"],
            has_ack,
            steer_norm,
        ], dtype=np.float32)
        return np.clip(vec, 0.0, 5.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "display_name": self.display_name,
            "provenance": self.provenance,
            "body_radius_m": self.body_radius_m,
            "boundary_clearance_m": self.boundary_clearance_m,
            "obstacle_clearance_m": self.obstacle_clearance_m,
            "working_width_m": self.working_width_m,
            "max_straight_speed_m_s": self.max_straight_speed_m_s,
            "max_turning_speed_m_s": self.max_turning_speed_m_s,
            "min_turning_radius_m": self.min_turning_radius_m,
            "max_curvature_inv_m": round(self.max_curvature_inv_m, 6),
            "max_curvature_rate_inv_m2": self.max_curvature_rate_inv_m2,
            "implied_max_lateral_accel_m_s2": round(self.implied_max_lateral_accel_m_s2, 6),
            "max_angular_velocity_rad_s": self.max_angular_velocity_rad_s,
            "max_linear_accel_m_s2": self.max_linear_accel_m_s2,
            "max_linear_decel_m_s2": self.max_linear_decel_m_s2,
            "max_angular_accel_rad_s2": self.max_angular_accel_rad_s2,
            "forward_only": self.forward_only,
            "allow_in_place_rotation": self.allow_in_place_rotation,
            "tool_always_active": self.tool_always_active,
            "wheelbase_m": self.wheelbase_m,
            "max_steering_angle_rad": self.max_steering_angle_rad,
            "max_steering_rate_rad_s": self.max_steering_rate_rad_s,
            "implement_collision_status": self.implement_collision_status,
            "energy_enabled": self.energy.enabled,
            "energy_status": self.energy.status,
            "battery_capacity_wh": self.energy.battery_capacity_wh,
            "notes": self.notes,
        }


def get_reference_robot() -> RobotSpec:
    """Robot 1: Exact repository-registered Bunker agricultural reference robot."""
    return RobotSpec(
        robot_id="robot_1_reference",
        display_name="Reference Bunker Ag-Robot (Registered)",
        provenance="registered_reference",
        body_radius_m=1.0,
        boundary_clearance_m=0.5,
        obstacle_clearance_m=0.5,
        working_width_m=6.0,
        max_straight_speed_m_s=1.5,
        max_turning_speed_m_s=0.5,
        min_turning_radius_m=3.0,
        max_curvature_rate_inv_m2=0.1,
        max_angular_velocity_rad_s=1.0,
        max_linear_accel_m_s2=0.5,
        max_linear_decel_m_s2=0.5,
        max_angular_accel_rad_s2=2.0,
        forward_only=True,
        allow_in_place_rotation=False,
        tool_always_active=True,
        wheelbase_m=None,
        max_steering_angle_rad=None,
        max_steering_rate_rad_s=None,
        implement_collision_status="unverified",
        energy=EnergyModelSpec(enabled=False, status="unverified_unavailable"),
        notes=(
            "Exact union of coverage_benchmark.contracts.Config, field_map_manager/config/field.yaml, "
            "and bunker_nav2/config/limits.yaml. Physical implement collision envelope and hardware "
            "battery/energy parameters are unavailable in the repository."
        ),
    )


def get_heterogeneous_robots() -> dict[str, RobotSpec]:
    """Return the 4 heterogeneous robot configurations (1 registered + 3 synthetic experimental)."""
    r1 = get_reference_robot()
    r2 = RobotSpec(
        robot_id="robot_2_compact_agile",
        display_name="Compact Agile Robot (Synthetic Experimental)",
        provenance="synthetic_experimental",
        body_radius_m=0.60,
        boundary_clearance_m=0.35,
        obstacle_clearance_m=0.35,
        working_width_m=3.0,
        max_straight_speed_m_s=2.0,
        max_turning_speed_m_s=0.8,
        min_turning_radius_m=1.5,
        max_curvature_rate_inv_m2=0.25,
        max_angular_velocity_rad_s=1.5,
        max_linear_accel_m_s2=0.8,
        max_linear_decel_m_s2=0.9,
        max_angular_accel_rad_s2=3.0,
        forward_only=True,
        allow_in_place_rotation=False,
        tool_always_active=True,
        implement_collision_status="unverified",
        energy=EnergyModelSpec(enabled=False, status="synthetic_optional_disabled_for_ppo"),
        notes="Synthetic experimental compact agile robot (NOT verified commercial hardware).",
    )
    r3 = RobotSpec(
        robot_id="robot_3_wide_heavy",
        display_name="Wide Heavy Carrier (Synthetic Experimental)",
        provenance="synthetic_experimental",
        body_radius_m=1.40,
        boundary_clearance_m=0.60,
        obstacle_clearance_m=0.60,
        working_width_m=9.0,
        max_straight_speed_m_s=1.2,
        max_turning_speed_m_s=0.4,
        min_turning_radius_m=5.0,
        max_curvature_rate_inv_m2=0.05,
        max_angular_velocity_rad_s=0.6,
        max_linear_accel_m_s2=0.3,
        max_linear_decel_m_s2=0.4,
        max_angular_accel_rad_s2=1.0,
        forward_only=True,
        allow_in_place_rotation=False,
        tool_always_active=True,
        implement_collision_status="unverified",
        energy=EnergyModelSpec(enabled=False, status="synthetic_optional_disabled_for_ppo"),
        notes="Synthetic experimental wide heavy agricultural robot (NOT verified commercial hardware).",
    )
    wheelbase = 2.2
    delta_max = math.atan(wheelbase / 4.0)
    r4 = RobotSpec(
        robot_id="robot_4_fast_ackermann",
        display_name="Fast Ackermann Robot (Synthetic Experimental)",
        provenance="synthetic_experimental",
        body_radius_m=0.90,
        boundary_clearance_m=0.50,
        obstacle_clearance_m=0.50,
        working_width_m=4.5,
        max_straight_speed_m_s=2.5,
        max_turning_speed_m_s=0.7,
        min_turning_radius_m=4.0,
        max_curvature_rate_inv_m2=0.08,
        max_angular_velocity_rad_s=0.9,
        max_linear_accel_m_s2=0.6,
        max_linear_decel_m_s2=0.8,
        max_angular_accel_rad_s2=1.5,
        forward_only=True,
        allow_in_place_rotation=False,
        tool_always_active=True,
        wheelbase_m=wheelbase,
        max_steering_angle_rad=delta_max,
        max_steering_rate_rad_s=0.55,
        implement_collision_status="unverified",
        energy=EnergyModelSpec(enabled=False, status="synthetic_optional_disabled_for_ppo"),
        notes="Synthetic experimental fast Ackermann robot (NOT verified commercial hardware).",
    )
    return {r.robot_id: r for r in (r1, r2, r3, r4)}


def get_robot_by_id(robot_id: str) -> RobotSpec:
    robots = get_heterogeneous_robots()
    if robot_id not in robots:
        raise KeyError(f"Unknown robot_id {robot_id!r}; expected one of {sorted(robots.keys())}")
    return robots[robot_id]


def sample_synthetic_robot_spec(rng: np.random.Generator) -> RobotSpec:
    """Sample a physically consistent synthetic RobotSpec within documented parameter ranges (Stage 7+)."""
    w = float(rng.uniform(3.0, 9.0))
    r_body = float(rng.uniform(0.6, 1.4))
    clr = float(rng.uniform(0.35, 0.65))
    v_max = float(rng.uniform(1.1, 2.5))
    v_turn = float(rng.uniform(0.35, min(0.9, 0.65 * v_max)))
    r_min = float(rng.uniform(1.5, 5.0))
    sigma_max = float(rng.uniform(0.05, 0.25))
    a_max = float(rng.uniform(0.3, 0.85))
    d_max = float(rng.uniform(0.4, 0.95))
    omega_max = max(float(rng.uniform(0.6, 1.5)), v_turn / r_min + 0.1)
    alpha_max = float(rng.uniform(1.0, 3.0))
    return RobotSpec(
        robot_id=f"sampled_w{w:.1f}_R{r_min:.1f}",
        display_name="Sampled Synthetic Robot",
        provenance="sampled_synthetic",
        body_radius_m=r_body,
        boundary_clearance_m=clr,
        obstacle_clearance_m=clr,
        working_width_m=w,
        max_straight_speed_m_s=v_max,
        max_turning_speed_m_s=v_turn,
        min_turning_radius_m=r_min,
        max_curvature_rate_inv_m2=sigma_max,
        max_angular_velocity_rad_s=omega_max,
        max_linear_accel_m_s2=a_max,
        max_linear_decel_m_s2=d_max,
        max_angular_accel_rad_s2=alpha_max,
        implement_collision_status="unverified",
        notes="Procedurally sampled synthetic robot for domain randomization.",
    )
