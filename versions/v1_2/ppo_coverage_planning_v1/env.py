"""Gymnasium-native PPO agricultural coverage planning environment (PPOCoverageEnv).

Combines:
  - Continuous-curvature RK4 vehicle dynamics (dynamics.py)
  - Families A-D field geometry & zero-tunneling swept collision checking (field_world.py)
  - Step-invariant incremental raster coverage & local/global maps (coverage_map.py)
  - Scale-consistent agricultural reward breakdown (rewards.py)
  - Fixed-field evaluation OR curriculum/dataset randomized episode resets (datasets.py, curriculum.py)

Observability classification:
  Partially Observable Markov Decision Process (POMDP): the policy receives an ego-aligned
  high-resolution local map (5x24x24 @ 36 m = 1.5 m/cell), an aspect-preserving global map
  (6x32x32, ego-aligned or world-aligned), 24 typed exterior/obstacle raycasts, normalized
  proprioception, 12-D robot capabilities, and 10-D task progress, without access to future
  actions, expert trajectories, swaths, or BCD decompositions.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import math
from typing import Any

import gymnasium as gym
import numpy as np

from .coverage_map import CoverageStepDelta, IncrementalCoverageMap
from .curriculum import CURRICULUM_STAGES, sample_task_for_stage
from .datasets import AgriculturalFieldDataset, FieldRecord
from .dynamics import ControlInterface, StepDiagnostics, VehicleSimulator
from .field_world import AgriculturalFieldWorld, CollisionMode, CollisionReport
from .rewards import RewardConfig, compute_step_reward
from .robot_specs import CAPABILITY_FEATURE_NAMES, RobotSpec, get_reference_robot


PROPRIO_FEATURE_NAMES: tuple[str, ...] = (
    "v_norm",
    "kappa_norm",
    "omega_norm",
    "v_allowed_norm",
    "sin_theta",
    "cos_theta",
    "prev_action_0",
    "prev_action_1",
    "any_saturation",
    "turning_speed_limited",
    "anticipatory_braking",
)

PROGRESS_FEATURE_NAMES: tuple[str, ...] = (
    "coverage_frac",
    "untreated_frac",
    "repeated_frac_scaled",
    "distance_norm_scaled",
    "elapsed_time_frac",
    "remaining_time_frac",
    "time_since_new_cov_norm",
    "recent_cov_rate_norm",
    "field_diag_norm",
    "field_aspect_ratio",
)


@dataclass(frozen=True)
class ObservationConfig:
    """Documented dimensions, resolutions, and toggles for the Dict observation space."""
    include_local_map: bool = True
    include_global_map: bool = True
    include_global_untreated_cue: bool = False  # 11-D ego-centric untreated sector histogram + nearest bearing/dist
    local_map_size: int = 24
    local_map_extent_m: float = 36.0          # 1.5 m/cell (resolves compact 3 m robot & narrow corridors)
    global_map_size: int = 32
    global_map_frame: str = "ego_aligned"     # "ego_aligned", "robot_centered", or "world"
    n_lidar_rays: int = 24
    lidar_max_range_m: float = 36.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PPOEnvConfig:
    """Complete episode, physics, timing, collision, observation, and reward configuration."""
    dt_action_s: float = 0.50                 # PPO decision interval (default 2 Hz; tested 0.2/0.5/1.0/2.0 s)
    dt_phys_s: float = 0.02                   # Continuous RK4 physics sub-step (50 Hz)
    coverage_resolution_m: float = 0.50       # Underlying treatment accounting raster resolution
    control_interface: ControlInterface = ControlInterface.TARGET_SPEED_CURVATURE
    collision_mode: CollisionMode = CollisionMode.TERMINATE
    coverage_target_pct: float = 99.0         # Episode completion target (curriculum may lower; 99% always tracked)
    time_budget_multiplier: float = 3.5       # max_sim_time_s = max(min_episode_time_s, mult * T_scale)
    min_episode_time_s: float = 150.0
    max_episode_time_cap_s: float = 3600.0    # Adaptive cap raised from 1800s -> 3600s so 5,000-9,000 m2 fields get full 3.5*T_scale
    max_stagnation_time_s: float | None = None  # Optional exploit-proof stagnation truncation (None in strict eval)
    randomize_start_pose: bool = True
    start_clearance_margin_m: float = 0.35
    meaningful_new_area_m2: float = 0.25      # Threshold for resetting time_since_new_coverage
    obs: ObservationConfig = field(default_factory=ObservationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["control_interface"] = self.control_interface.value
        d["collision_mode"] = self.collision_mode.value
        return d


class PPOCoverageEnv(gym.Env):
    """Physically constrained end-to-end agricultural coverage planning Gymnasium environment."""

    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 10}

    def __init__(
        self,
        *,
        env_config: PPOEnvConfig | None = None,
        world: AgriculturalFieldWorld | None = None,
        robot_spec: RobotSpec | None = None,
        dataset: AgriculturalFieldDataset | None = None,
        curriculum_stage: int | None = None,
        split: str = "train",
        fixed_robot: RobotSpec | None = None,
        allow_sampled_robots: bool = False,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.cfg = env_config or PPOEnvConfig()
        self.dataset = dataset
        self.curriculum_stage = curriculum_stage
        self.split = split
        self.fixed_robot = fixed_robot
        self.allow_sampled_robots = allow_sampled_robots
        self.render_mode = render_mode

        self._fixed_world = world
        self._fixed_spec = robot_spec
        self._rng = np.random.default_rng(0)

        # Action space: normalized [-1, 1]^d
        act_dim = self.cfg.control_interface.action_dim
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)

        # Documented finite Dict observation space
        oc = self.cfg.obs
        obs_spaces: dict[str, gym.Space] = {
            "proprio": gym.spaces.Box(low=-5.0, high=5.0, shape=(len(PROPRIO_FEATURE_NAMES),), dtype=np.float32),
            "capabilities": gym.spaces.Box(low=0.0, high=5.0, shape=(len(CAPABILITY_FEATURE_NAMES),), dtype=np.float32),
            "progress": gym.spaces.Box(low=0.0, high=2.0, shape=(len(PROGRESS_FEATURE_NAMES),), dtype=np.float32),
            "obstacles": gym.spaces.Box(low=-1.0, high=2.0, shape=(2 * oc.n_lidar_rays + 4,), dtype=np.float32),
        }
        if oc.include_local_map:
            obs_spaces["local_map"] = gym.spaces.Box(
                low=0.0, high=1.0, shape=(5, oc.local_map_size, oc.local_map_size), dtype=np.float32
            )
        if oc.include_global_map:
            obs_spaces["global_map"] = gym.spaces.Box(
                low=0.0, high=1.0, shape=(6, oc.global_map_size, oc.global_map_size), dtype=np.float32
            )
        if oc.include_global_untreated_cue:
            obs_spaces["global_untreated"] = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(11,), dtype=np.float32
            )
        self.observation_space = gym.spaces.Dict(obs_spaces)

        # Cache worlds/coverage grids for fixed-field or repeated real fields to avoid re-rasterizing masks
        self._world_cache: dict[str, AgriculturalFieldWorld] = {}

        # Active episode objects (initialized on first reset or immediately if fixed world supplied)
        self.world: AgriculturalFieldWorld | None = None
        self.spec: RobotSpec = robot_spec or fixed_robot or get_reference_robot()
        self.sim: VehicleSimulator | None = None
        self.coverage: IncrementalCoverageMap | None = None
        self._field_record: FieldRecord | None = None

        # Episode counters & telemetry
        self._done = True
        self._step_count = 0
        self._max_sim_time_s = self.cfg.min_episode_time_s
        self._max_steps = int(math.ceil(self._max_sim_time_s / self.cfg.dt_action_s))
        self._l_scale_m = 100.0
        self._t_scale_s = 100.0
        self._stage_target_pct = self.cfg.coverage_target_pct
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._last_diag: StepDiagnostics | None = None
        self._last_coll: CollisionReport | None = None
        self._time_since_new_cov_s = 0.0
        self._recent_new_areas: deque[tuple[float, float]] = deque(maxlen=20)
        self._trajectory_xy: list[tuple[float, float]] = []
        self._trajectory_poses: list[tuple[float, float, float, float, float, float]] = []
        self._executed_controls: list[dict[str, float]] = []

        # Cumulative episode diagnostics for training monitoring & evaluation
        self._ep_reward = 0.0
        self._ep_reward_terms: dict[str, float] = {}
        self._ep_body_collisions = 0
        self._ep_ext_collisions = 0
        self._ep_obs_collisions = 0
        self._ep_clearance_violations = 0
        self._ep_saturation_steps = 0
        self._ep_safety_filter_steps = 0
        self._ep_constraint_violations = 0
        self._ep_unproductive_dist_m = 0.0
        self._ep_time_without_new_cov_s = 0.0
        self._ep_stationary_steps = 0
        self._ep_treated_kappa_sum = 0.0
        self._ep_treated_kappa_steps = 0

    def set_curriculum_stage(self, stage_id: int) -> None:
        if stage_id not in CURRICULUM_STAGES:
            raise ValueError(f"Unknown curriculum stage {stage_id}")
        self.curriculum_stage = int(stage_id)

    def _setup_episode_world_and_robot(self, opts: dict[str, Any]) -> None:
        if "world" in opts and opts["world"] is not None:
            world = opts["world"]
            spec = opts.get("robot_spec", self._fixed_spec or self.fixed_robot or get_reference_robot())
            self._field_record = None
            target_pct = float(opts.get("coverage_target_pct", self.cfg.coverage_target_pct))
            time_mult = float(opts.get("time_budget_multiplier", self.cfg.time_budget_multiplier))
            min_t = float(opts.get("min_episode_time_s", self.cfg.min_episode_time_s))
        elif self._fixed_world is not None:
            world = self._fixed_world
            spec = opts.get("robot_spec", self._fixed_spec or self.fixed_robot or get_reference_robot())
            self._field_record = None
            target_pct = float(opts.get("coverage_target_pct", self.cfg.coverage_target_pct))
            time_mult = float(opts.get("time_budget_multiplier", self.cfg.time_budget_multiplier))
            min_t = float(opts.get("min_episode_time_s", self.cfg.min_episode_time_s))
        else:
            stage = int(opts.get("curriculum_stage", self.curriculum_stage if self.curriculum_stage is not None else 1))
            scfg = CURRICULUM_STAGES[stage]
            rec, spec = sample_task_for_stage(
                stage,
                self._rng,
                dataset=self.dataset,
                split=self.split,
                fixed_robot=opts.get("robot_spec", self.fixed_robot),
                allow_sampled_robots=self.allow_sampled_robots,
            )
            self._field_record = rec
            if rec.source == "real_eu" and rec.field_id in self._world_cache:
                world = self._world_cache[rec.field_id]
            else:
                world = rec.to_world(sdf_resolution_m=0.5)
                if rec.source == "real_eu" and len(self._world_cache) < 64:
                    self._world_cache[rec.field_id] = world
            target_pct = float(opts.get("coverage_target_pct", scfg.coverage_target_pct))
            time_mult = float(opts.get("time_budget_multiplier", scfg.time_budget_multiplier))
            min_t = float(opts.get("min_episode_time_s", scfg.min_episode_time_s))

        self.world = world
        self.spec = spec
        self._stage_target_pct = target_pct
        self._l_scale_m = max(1.0, world.source_area_m2 / spec.working_width_m)
        self._t_scale_s = max(1.0, self._l_scale_m / spec.max_straight_speed_m_s)
        raw_budget = max(min_t, time_mult * self._t_scale_s)
        self._max_sim_time_s = float(opts.get("max_sim_time_s", min(self.cfg.max_episode_time_cap_s, raw_budget)))
        self._max_steps = int(opts.get("max_steps", math.ceil(self._max_sim_time_s / self.cfg.dt_action_s)))

        self.sim = VehicleSimulator(spec, dt_phys=self.cfg.dt_phys_s, interface=self.cfg.control_interface)
        nav_poly = world.navigable_domain(spec, extra_margin_m=0.0)
        # Reuse existing coverage map buffer when world & resolution haven't changed (huge speed win!)
        if (
            self.coverage is not None
            and self.coverage.polygon is world.polygon
            and abs(self.coverage.resolution_m - self.cfg.coverage_resolution_m) < 1e-9
        ):
            self.coverage.reset()
            self.coverage.set_navigable_polygon(nav_poly)
        else:
            self.coverage = IncrementalCoverageMap(
                world.polygon,
                resolution_m=self.cfg.coverage_resolution_m,
                pad_m=max(6.0, spec.working_width_m + 1.0),
                navigable_polygon=nav_poly,
            )

    def _build_observation(self) -> dict[str, np.ndarray]:
        assert self.sim is not None and self.world is not None and self.coverage is not None
        st = self.sim.state
        s = self.spec
        oc = self.cfg.obs
        kmax = s.max_curvature_inv_m
        v_allow = s.allowed_speed(st.kappa)
        diag = self._last_diag

        proprio = np.array([
            st.v / max(1e-6, s.max_straight_speed_m_s),
            st.kappa / max(1e-6, kmax),
            st.omega / max(1e-6, s.max_angular_velocity_rad_s),
            v_allow / max(1e-6, s.max_straight_speed_m_s),
            math.sin(st.theta),
            math.cos(st.theta),
            float(self._prev_action[0]),
            float(self._prev_action[1]),
            1.0 if (diag is not None and diag.any_saturation) else 0.0,
            1.0 if (diag is not None and diag.turning_speed_limited) else 0.0,
            1.0 if (diag is not None and diag.anticipatory_braking_active) else 0.0,
        ], dtype=np.float32)

        capabilities = s.to_capability_vector()

        snap = self.coverage.snapshot()
        recent_dt = sum(dt for _, dt in self._recent_new_areas)
        recent_da = sum(da for da, _ in self._recent_new_areas)
        recent_rate = (recent_da / recent_dt) if recent_dt > 1e-6 else 0.0
        max_rate = max(1e-6, s.working_width_m * s.max_straight_speed_m_s)
        aspect = min(self.world.width_m, self.world.height_m) / max(1e-6, max(self.world.width_m, self.world.height_m))

        progress = np.array([
            min(1.0, snap.coverage_pct / 100.0),
            max(0.0, min(1.0, snap.untreated_area_m2 / max(1.0, snap.source_area_m2))),
            min(5.0, snap.repeated_treatment_pct / 100.0) / 5.0,
            min(5.0, st.s / max(1.0, self._l_scale_m)) / 5.0,
            min(1.0, st.t / max(1e-6, self._max_sim_time_s)),
            max(0.0, 1.0 - st.t / max(1e-6, self._max_sim_time_s)),
            min(1.0, self._time_since_new_cov_s / 30.0),
            min(2.0, recent_rate / max_rate) / 2.0,
            min(2.0, self.world.bounding_diag_m / 200.0) / 2.0,
            min(1.0, max(0.0, aspect)),
        ], dtype=np.float32)

        d_ext_rays, d_obs_rays = self.world.cast_typed_lidar_rays(
            st.x, st.y, st.theta, n_rays=oc.n_lidar_rays, max_range_m=oc.lidar_max_range_m
        )
        inv_r = np.float32(1.0 / oc.lidar_max_range_m)
        d_ext_min = float(np.min(d_ext_rays))
        d_obs_min = float(np.min(d_obs_rays))
        body_clr_ext = np.clip((d_ext_min - s.body_radius_m) / 10.0, -0.5, 1.0)
        body_clr_obs = np.clip((d_obs_min - s.body_radius_m) / 10.0, -0.5, 1.0)
        margin_req = min(
            d_ext_min - s.exterior_center_clearance_m,
            d_obs_min - s.obstacle_center_clearance_m,
        )
        margin_norm = np.clip(margin_req / 5.0, -0.5, 1.0)
        req_clr_norm = np.clip(max(s.boundary_clearance_m, s.obstacle_clearance_m) / 2.0, 0.0, 1.0)
        obstacles = np.concatenate([
            d_ext_rays * inv_r,
            d_obs_rays * inv_r,
            np.array([body_clr_ext, body_clr_obs, margin_norm, req_clr_norm], dtype=np.float32),
        ]).astype(np.float32)

        obs: dict[str, np.ndarray] = {
            "proprio": np.clip(proprio, -5.0, 5.0),
            "capabilities": capabilities,
            "progress": np.clip(progress, 0.0, 2.0),
            "obstacles": np.clip(obstacles, -1.0, 2.0),
        }
        if oc.include_local_map:
            obs["local_map"] = self.coverage.extract_local_map(
                st.x, st.y, st.theta,
                patch_size=oc.local_map_size,
                patch_extent_m=oc.local_map_extent_m,
            )
        if oc.include_global_map:
            obs["global_map"] = self.coverage.extract_global_map(
                st.x, st.y, st.theta,
                global_size=oc.global_map_size,
                frame=oc.global_map_frame,
            )
        if oc.include_global_untreated_cue:
            obs["global_untreated"] = np.clip(
                self.coverage.extract_global_untreated_cue(st.x, st.y, st.theta), -1.0, 1.0
            )
        return obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        opts = options or {}

        self._setup_episode_world_and_robot(opts)
        assert self.world is not None and self.sim is not None and self.coverage is not None

        if "initial_pose" in opts and opts["initial_pose"] is not None:
            ix, iy, ith = opts["initial_pose"]
            if not self.world.is_point_valid_start(float(ix), float(iy), self.spec, margin_m=0.0):
                raise ValueError(f"Initial pose ({ix}, {iy}) violates physical clearance in {self.world.field_id}")
        elif self.cfg.randomize_start_pose and opts.get("randomize_start_pose", True):
            ix, iy, ith = self.world.sample_valid_start_pose(
                self.spec, self._rng, margin_m=self.cfg.start_clearance_margin_m
            )
        else:
            nav = self.world.navigable_domain(self.spec, extra_margin_m=self.cfg.start_clearance_margin_m)
            rp = nav.representative_point() if not nav.is_empty else self.world.polygon.representative_point()
            ix, iy, ith = float(rp.x), float(rp.y), 0.0

        iv = float(opts.get("initial_v", 0.0))
        ik = float(opts.get("initial_kappa", 0.0))
        self.sim.reset(ix, iy, ith, v=iv, kappa=ik)
        if "initial_coverage_mask" in opts and opts["initial_coverage_mask"] is not None:
            self.coverage.apply_initial_coverage_mask(opts["initial_coverage_mask"])

        self._done = False
        self._step_count = 0
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._last_diag = None
        self._last_coll = None
        self._time_since_new_cov_s = 0.0
        self._recent_new_areas.clear()
        st = self.sim.state
        self._trajectory_xy = [(st.x, st.y)]
        self._trajectory_poses = [(st.t, st.x, st.y, st.theta, st.v, st.kappa)]
        self._executed_controls = []

        self._ep_reward = 0.0
        self._ep_reward_terms = {}
        self._ep_body_collisions = 0
        self._ep_ext_collisions = 0
        self._ep_obs_collisions = 0
        self._ep_clearance_violations = 0
        self._ep_saturation_steps = 0
        self._ep_safety_filter_steps = 0
        self._ep_constraint_violations = 0
        self._ep_unproductive_dist_m = 0.0
        self._ep_time_without_new_cov_s = 0.0
        self._ep_stationary_steps = 0
        self._ep_treated_kappa_sum = 0.0
        self._ep_treated_kappa_steps = 0

        obs = self._build_observation()
        info = {
            "field_id": self.world.field_id,
            "family": self.world.family,
            "robot_id": self.spec.robot_id,
            "source_area_m2": self.world.source_area_m2,
            "max_sim_time_s": self._max_sim_time_s,
            "max_steps": self._max_steps,
            "stage_target_pct": self._stage_target_pct,
            "implement_collision_status": self.spec.implement_collision_status,
        }
        return obs, info

    def step(
        self,
        action: np.ndarray | tuple[float, ...] | list[float],
        *,
        physical_command: bool = False,
        record_trajectory: bool = True,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        assert self.sim is not None and self.world is not None and self.coverage is not None
        if self._done:
            obs = self._build_observation()
            return obs, 0.0, True, False, {"warning": "step_after_done"}

        act_arr = np.asarray(action, dtype=np.float32).ravel()
        prev_state = self.sim.state.copy()

        # Execute continuous RK4 physics across dt_action_s
        new_state, diag = self.sim.step(
            act_arr,
            self.cfg.dt_action_s,
            normalized_action=(not physical_command),
            record_substeps=True,
        )

        # Audit physical constraints on every sub-step
        s = self.spec
        constr_viol = (
            diag.min_v_substep < -1e-6
            or diag.max_v_substep > s.max_straight_speed_m_s + 1e-5
            or diag.max_abs_kappa_substep > s.max_curvature_inv_m + 1e-5
            or diag.max_abs_omega_substep > s.max_angular_velocity_rad_s + 1e-5
            or diag.max_abs_spatial_rate_substep > s.max_curvature_rate_inv_m2 + 1e-5
            or diag.max_abs_accel_substep > max(s.max_linear_accel_m_s2, s.max_linear_decel_m_s2) + 1e-5
            or diag.max_turning_speed_excess_m_s > 1e-4
        )
        if constr_viol:
            self._ep_constraint_violations += 1

        # Continuous zero-tunneling swept collision & clearance check
        coll = self.world.check_motion_substeps(diag.substep_states, self.spec)

        # Collision handling semantics (NEVER reward unexecuted/post-collision motion!)
        executed_substeps = diag.substep_states
        executed_dist = diag.distance_traveled_m
        safety_filter_intervened = False
        unphysical_stop_event = False
        terminate_due_to_safety = False

        mode = self.cfg.collision_mode
        if not coll.safe:
            if mode == CollisionMode.TERMINATE:
                terminate_due_to_safety = True
                # Execute coverage ONLY on the certified safe prefix before the violation!
                safe_dt = max(0.0, coll.safe_prefix_fraction * self.cfg.dt_action_s * 0.98)
                self.sim.state = prev_state
                if safe_dt > 1e-4:
                    new_state, diag_safe = self.sim.step(
                        act_arr, safe_dt, normalized_action=(not physical_command), record_substeps=True
                    )
                    executed_substeps = diag_safe.substep_states
                    executed_dist = diag_safe.distance_traveled_m
                else:
                    executed_substeps = []
                    executed_dist = 0.0
                    self.sim.state.t = prev_state.t + self.cfg.dt_action_s
                    new_state = self.sim.state.copy()
            elif mode == CollisionMode.TERMINATE_BODY_ONLY:
                if coll.body_collision:
                    terminate_due_to_safety = True
                    safe_dt = max(0.0, coll.body_safe_prefix_fraction * self.cfg.dt_action_s * 0.98)
                    self.sim.state = prev_state
                    if safe_dt > 1e-4:
                        new_state, diag_safe = self.sim.step(
                            act_arr, safe_dt, normalized_action=(not physical_command), record_substeps=True
                        )
                        executed_substeps = diag_safe.substep_states
                        executed_dist = diag_safe.distance_traveled_m
                    else:
                        executed_substeps = []
                        executed_dist = 0.0
                        self.sim.state.t = prev_state.t + self.cfg.dt_action_s
                        new_state = self.sim.state.copy()
            elif mode == CollisionMode.REJECT_ACTION:
                safety_filter_intervened = True
                unphysical_stop_event = True
                self.sim.state = prev_state
                self.sim.state.t += self.cfg.dt_action_s
                new_state = self.sim.state.copy()
                executed_substeps = []
                executed_dist = 0.0
            elif mode == CollisionMode.STOP_AT_PREFIX:
                safety_filter_intervened = True
                unphysical_stop_event = True
                safe_dt = max(0.0, coll.safe_prefix_fraction * self.cfg.dt_action_s * 0.98)
                self.sim.state = prev_state
                if safe_dt > 1e-4:
                    new_state, diag_safe = self.sim.step(
                        act_arr, safe_dt, normalized_action=(not physical_command), record_substeps=True
                    )
                    executed_substeps = diag_safe.substep_states
                    executed_dist = diag_safe.distance_traveled_m
                else:
                    executed_substeps = []
                    executed_dist = 0.0
                self.sim.state.v = 0.0
                self.sim.state.t = prev_state.t + self.cfg.dt_action_s
                new_state = self.sim.state.copy()

        # Incremental coverage update on certified executed motion ONLY
        cov_pct_before = float(self.coverage.snapshot().coverage_pct)
        if executed_substeps:
            cov_delta = self.coverage.update_from_substeps(
                executed_substeps, self.spec.working_width_m, tool_active=self.spec.tool_always_active
            )
        else:
            cov_delta = CoverageStepDelta(0.0, 0.0, 0.0, 0.0, 0.0, cov_pct_before, self.coverage.snapshot().untreated_area_m2)

        # Update productivity & safety counters
        self._step_count += 1
        if coll.body_collision:
            self._ep_body_collisions += 1
        if coll.exterior_collision:
            self._ep_ext_collisions += 1
        if coll.obstacle_collision:
            self._ep_obs_collisions += 1
        if coll.clearance_breach:
            self._ep_clearance_violations += 1
        if diag.any_saturation:
            self._ep_saturation_steps += 1
        if safety_filter_intervened:
            self._ep_safety_filter_steps += 1

        if cov_delta.newly_treated_m2 >= self.cfg.meaningful_new_area_m2:
            self._time_since_new_cov_s = 0.0
        else:
            self._time_since_new_cov_s += self.cfg.dt_action_s
            self._ep_time_without_new_cov_s += self.cfg.dt_action_s
            self._ep_unproductive_dist_m += executed_dist
        self._recent_new_areas.append((cov_delta.newly_treated_m2, self.cfg.dt_action_s))

        if record_trajectory:
            self._trajectory_xy.append((new_state.x, new_state.y))
            self._trajectory_poses.append((new_state.t, new_state.x, new_state.y, new_state.theta, new_state.v, new_state.kappa))
            self._executed_controls.append({
                "t": new_state.t,
                "req_0": diag.requested_cmd[0],
                "req_1": diag.requested_cmd[1],
                "exec_v": new_state.v,
                "exec_kappa": new_state.kappa,
                "exec_accel": diag.executed_accel_m_s2,
                "exec_sigma": diag.executed_spatial_rate_inv_m2,
            })

        # Store previous normalized action for proprioception
        if self.cfg.control_interface == ControlInterface.CURVATURE_ONLY:
            self._prev_action = np.array([1.0, float(np.clip(act_arr[0], -1.0, 1.0))], dtype=np.float32)
        else:
            self._prev_action = np.clip(act_arr[:2], -1.0, 1.0).astype(np.float32)
        self._last_diag = diag
        self._last_coll = coll

        # Termination (natural) vs Truncation (time/step budget)
        terminated = False
        termination_reason = "running"
        safe_clean_so_far = (self._ep_body_collisions == 0) and (self._ep_clearance_violations == 0)
        coverage_completed_safely = False

        if terminate_due_to_safety:
            terminated = True
            termination_reason = coll.collision_type
        elif cov_delta.coverage_pct_after >= self._stage_target_pct and safe_clean_so_far:
            terminated = True
            termination_reason = "coverage_target_reached"
            coverage_completed_safely = True

        truncated = False
        stagnation_remaining_s = 0.0
        if not terminated:
            if self._step_count >= self._max_steps:
                truncated = True
                termination_reason = "max_steps_reached"
            elif new_state.t >= self._max_sim_time_s - 1e-9:
                truncated = True
                termination_reason = "max_sim_time_reached"
            elif (
                self.cfg.max_stagnation_time_s is not None
                and self._time_since_new_cov_s >= self.cfg.max_stagnation_time_s - 1e-9
            ):
                truncated = True
                termination_reason = "stagnation_limit_reached"
                stagnation_remaining_s = max(0.0, self._max_sim_time_s - new_state.t)

        if terminated or truncated:
            self._done = True

        clearance_margin_above_req = min(
            coll.min_exterior_dist_m - self.spec.exterior_center_clearance_m,
            coll.min_obstacle_dist_m - self.spec.obstacle_center_clearance_m,
        )
        avg_speed_m_s = executed_dist / max(1e-6, self.cfg.dt_action_s)
        executed_speed_frac = min(1.0, max(0.0, avg_speed_m_s / max(1e-6, self.spec.max_straight_speed_m_s)))
        if executed_substeps:
            mean_abs_k = sum(abs(float(st_sub[4])) for st_sub in executed_substeps) / len(executed_substeps)
        else:
            mean_abs_k = abs(float(new_state.kappa))
        abs_kappa_frac = min(1.0, max(0.0, mean_abs_k / max(1e-6, self.spec.max_curvature_inv_m)))
        if cov_delta.newly_treated_m2 < self.cfg.meaningful_new_area_m2 and executed_dist > 0.05:
            self._ep_treated_kappa_sum += abs_kappa_frac
            self._ep_treated_kappa_steps += 1
        if new_state.v < 0.05:
            self._ep_stationary_steps += 1

        reward, r_terms = compute_step_reward(
            self.cfg.reward,
            field_area_m2=self.world.source_area_m2,
            l_scale_m=self._l_scale_m,
            t_scale_s=self._t_scale_s,
            newly_treated_m2=cov_delta.newly_treated_m2,
            repeated_treated_m2=cov_delta.repeated_treated_m2,
            distance_traveled_m=executed_dist,
            dt_elapsed_s=self.cfg.dt_action_s,
            body_collision=(coll.body_collision and not safety_filter_intervened),
            clearance_breach=(coll.clearance_breach and not safety_filter_intervened),
            safety_filter_intervened=safety_filter_intervened,
            new_obstacle_intrusion_m2=cov_delta.new_obstacle_intrusion_m2,
            clearance_margin_above_req_m=clearance_margin_above_req,
            time_since_new_coverage_s=self._time_since_new_cov_s,
            coverage_completed_safely=coverage_completed_safely,
            executed_speed_frac=executed_speed_frac,
            stagnation_truncated_remaining_s=stagnation_remaining_s,
            abs_kappa_frac=abs_kappa_frac,
            coverage_pct_before=cov_pct_before,
            coverage_pct_after=float(cov_delta.coverage_pct_after),
            safe_clean_so_far=safe_clean_so_far,
        )

        self._ep_reward += reward
        for k, v in r_terms.items():
            self._ep_reward_terms[k] = self._ep_reward_terms.get(k, 0.0) + v

        obs = self._build_observation()
        snap = self.coverage.snapshot()

        info: dict[str, Any] = {
            "step": self._step_count,
            "field_id": self.world.field_id,
            "family": self.world.family,
            "robot_id": self.spec.robot_id,
            "termination_reason": termination_reason,
            "coverage_pct": snap.coverage_pct,
            "repeated_treatment_m2": snap.repeated_treatment_m2,
            "repeated_treatment_pct": snap.repeated_treatment_pct,
            "obstacle_treatment_m2": snap.obstacle_treatment_m2,
            "outside_field_treatment_m2": snap.outside_field_treatment_m2,
            "distance_m": new_state.s,
            "operating_time_s": new_state.t,
            "body_collision": coll.body_collision,
            "clearance_breach": coll.clearance_breach,
            "collision_type": coll.collision_type,
            "safety_filter_intervened": safety_filter_intervened,
            "unphysical_stop_event": unphysical_stop_event,
            "any_saturation": diag.any_saturation,
            "constraint_violation": constr_viol,
            "requested_cmd": diag.requested_cmd,
            "executed_v": new_state.v,
            "executed_kappa": new_state.kappa,
            "reward_terms": r_terms,
        }

        if terminated or truncated:
            largest_omission = self.coverage.largest_untreated_component_m2()
            clean_safe = (self._ep_body_collisions == 0) and (self._ep_clearance_violations == 0)
            info["episode_summary"] = {
                "field_id": self.world.field_id,
                "family": self.world.family,
                "robot_id": self.spec.robot_id,
                "source_area_m2": self.world.source_area_m2,
                "steps": self._step_count,
                "operating_time_s": new_state.t,
                "distance_m": new_state.s,
                "coverage_pct": snap.coverage_pct,
                "covered_area_m2": snap.covered_area_m2,
                "repeated_treatment_m2": snap.repeated_treatment_m2,
                "repeated_treatment_pct": snap.repeated_treatment_pct,
                "largest_omission_m2": largest_omission,
                "obstacle_treatment_m2": snap.obstacle_treatment_m2,
                "unproductive_dist_m": self._ep_unproductive_dist_m,
                "time_without_new_cov_s": self._ep_time_without_new_cov_s,
                "stationary_frac": self._ep_stationary_steps / max(1, self._step_count),
                "mean_abs_kappa_on_treated": self._ep_treated_kappa_sum / max(1, self._ep_treated_kappa_steps),
                "new_area_per_sec": snap.covered_area_m2 / max(1e-6, new_state.t),
                "new_area_per_m": snap.covered_area_m2 / max(1e-6, new_state.s),
                "episode_reward": self._ep_reward,
                "reward_terms_total": dict(self._ep_reward_terms),
                "termination_reason": termination_reason,
                "stage_target_pct": self._stage_target_pct,
                "stage_completed": bool(clean_safe and snap.coverage_pct >= self._stage_target_pct),
                "final_99_completed": bool(clean_safe and snap.coverage_pct >= 99.0),
                "body_collision": self._ep_body_collisions > 0,
                "exterior_collision": self._ep_ext_collisions > 0,
                "obstacle_collision": self._ep_obs_collisions > 0,
                "clearance_violation": self._ep_clearance_violations > 0,
                "saturation_rate": self._ep_saturation_steps / max(1, self._step_count),
                "safety_filter_rate": self._ep_safety_filter_steps / max(1, self._step_count),
                "constraint_violations": self._ep_constraint_violations,
            }

        return obs, float(reward), terminated, truncated, info

    def get_trajectory_xy(self) -> np.ndarray:
        return np.asarray(self._trajectory_xy, dtype=np.float64)

    def get_trajectory_poses(self) -> list[tuple[float, float, float, float, float, float]]:
        return list(self._trajectory_poses)

    def render(self) -> Any:
        if self.sim is None or self.world is None or self.coverage is None:
            return None
        if self.render_mode == "ansi":
            snap = self.coverage.snapshot()
            st = self.sim.state
            return (
                f"step={self._step_count} t={st.t:.1f}s pos=({st.x:.1f},{st.y:.1f}) "
                f"v={st.v:.2f}m/s k={st.kappa:+.3f} cov={snap.coverage_pct:.1f}%"
            )
        if self.render_mode == "rgb_array":
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle, Polygon as MplPolygon

            fig, ax = plt.subplots(figsize=(5.5, 5.2), dpi=100)
            ext = np.asarray(self.world.polygon.exterior.coords)[:, :2]
            ax.add_patch(MplPolygon(ext, closed=True, facecolor="#fef9c3", edgecolor="#0f172a", lw=1.6))

            cov_img = np.zeros((self.coverage.ny, self.coverage.nx, 4), dtype=np.float32)
            treated = self.coverage.covered & self.coverage.in_field
            rep = (self.coverage.treated_count >= 2) & self.coverage.in_field
            cov_img[treated & (~rep)] = (0.20, 0.78, 0.42, 0.70)
            cov_img[rep] = (0.95, 0.58, 0.15, 0.72)
            cov_img[self.coverage.in_obstacle] = (0.18, 0.22, 0.28, 0.92)
            ax.imshow(
                cov_img,
                origin="lower",
                extent=[
                    self.coverage.x0,
                    self.coverage.x0 + self.coverage.nx * self.coverage.resolution_m,
                    self.coverage.y0,
                    self.coverage.y0 + self.coverage.ny * self.coverage.resolution_m,
                ],
                zorder=2,
            )
            if len(self._trajectory_xy) >= 2:
                txy = np.asarray(self._trajectory_xy)
                ax.plot(txy[:, 0], txy[:, 1], color="#1d4ed8", lw=1.2, alpha=0.85, zorder=4)
                ax.scatter([txy[0, 0]], [txy[0, 1]], color="#16a34a", s=35, marker="o", zorder=6)
            st = self.sim.state
            nx_v, ny_v = -math.sin(st.theta), math.cos(st.theta)
            hw = 0.5 * self.spec.working_width_m
            ax.plot([st.x - hw * nx_v, st.x + hw * nx_v], [st.y - hw * ny_v, st.y + hw * ny_v],
                    color="#15803d", lw=3.2, zorder=6)
            ax.add_patch(Circle((st.x, st.y), self.spec.body_radius_m, facecolor="#1d4ed8", edgecolor="white", zorder=7))
            ax.arrow(st.x, st.y, 1.8 * math.cos(st.theta), 1.8 * math.sin(st.theta),
                     head_width=0.7, head_length=0.8, fc="white", ec="#0f172a", zorder=8)

            snap = self.coverage.snapshot()
            ax.set_title(
                f"{self.world.field_id} ({self.world.family}) | Cov {snap.coverage_pct:.1f}% | t={st.t:.1f}s | v={st.v:.2f} m/s",
                fontsize=9.5,
            )
            xmin, ymin, xmax, ymax = self.world.bounds
            pad = max(4.0, 0.08 * max(xmax - xmin, ymax - ymin))
            ax.set_xlim(xmin - pad, xmax + pad)
            ax.set_ylim(ymin - pad, ymax + pad)
            ax.set_aspect("equal")
            fig.tight_layout()
            fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            plt.close(fig)
            return rgb
        return None
