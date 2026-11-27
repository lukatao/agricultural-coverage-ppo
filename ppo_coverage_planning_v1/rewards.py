"""Mathematically aligned, scale-consistent agricultural coverage reward system.

All productive and cost terms are normalized by physical field & robot scales:
  - A_field = original agricultural field area excluding obstacles [m^2]
  - L_scale = A_field / working_width  [m]  (normalization scale, NOT a proven lower bound)
  - T_scale = L_scale / v_max          [s]  (characteristic nominal operating time scale)

Because every continuous term is proportional to an exact physical integral (dA_new, dA_repeat,
ds, dt), cumulative episode reward is invariant to the PPO action interval dt_action for the
same physical trajectory, and remains cleanly scaled across small and large fields.

No trajectory-following, swath-alignment, or boustrophedon-forcing rewards exist here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RewardConfig:
    """Configurable agricultural reward weights (every component logged separately)."""
    preset_name: str = "simple_v1"

    # Primary agricultural & efficiency terms (continuous physical integrals)
    alpha_new_area: float = 100.0        # +100.0 for covering 100% of A_field
    beta_repeat_area: float = 10.0       # -10.0 per A_field of avoidable repeated treatment
    gamma_distance: float = 5.0          # -5.0 per L_scale (= A_field / w) traveled
    eta_time: float = 5.0                # -5.0 per T_scale (= A_field / (w * v_max)) elapsed

    # Safety & collision event penalties (strictly worse than full timeout so early suicide never pays!)
    body_collision_penalty: float = 40.0
    clearance_breach_penalty: float = 35.0
    safety_filter_penalty: float = 2.0
    obstacle_treatment_penalty: float = 0.0   # Optional: penalty * (dA_obs / A_field)

    # Optional continuous clearance warning shaping (disabled in simple_v1; enabled in shaped_v1)
    clearance_shaping_weight: float = 0.0
    clearance_warning_margin_m: float = 1.5

    # Optional stagnation penalty after prolonged time without meaningful new coverage
    stagnation_penalty_weight: float = 0.0
    stagnation_grace_s: float = 15.0

    # Successful agricultural completion bonus (only when safe & target coverage achieved)
    success_bonus: float = 30.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_reward_preset(name: str) -> RewardConfig:
    if name == "simple_v1":
        return RewardConfig(preset_name="simple_v1")
    if name == "shaped_v1":
        return RewardConfig(
            preset_name="shaped_v1",
            alpha_new_area=100.0,
            beta_repeat_area=12.0,
            gamma_distance=5.0,
            eta_time=5.0,
            body_collision_penalty=40.0,
            clearance_breach_penalty=35.0,
            safety_filter_penalty=2.0,
            obstacle_treatment_penalty=10.0,
            clearance_shaping_weight=8.0,
            clearance_warning_margin_m=1.5,
            stagnation_penalty_weight=3.0,
            stagnation_grace_s=15.0,
            success_bonus=30.0,
        )
    raise KeyError(f"Unknown reward preset {name!r}; expected 'simple_v1' or 'shaped_v1'")


def compute_step_reward(
    cfg: RewardConfig,
    *,
    field_area_m2: float,
    l_scale_m: float,
    t_scale_s: float,
    newly_treated_m2: float,
    repeated_treated_m2: float,
    distance_traveled_m: float,
    dt_elapsed_s: float,
    body_collision: bool,
    clearance_breach: bool,
    safety_filter_intervened: bool,
    new_obstacle_intrusion_m2: float,
    clearance_margin_above_req_m: float,
    time_since_new_coverage_s: float,
    coverage_completed_safely: bool,
) -> tuple[float, dict[str, float]]:
    """Compute scalar reward and exact breakdown dict for one executed step."""
    inv_a = 1.0 / max(1.0, field_area_m2)
    inv_l = 1.0 / max(1.0, l_scale_m)
    inv_t = 1.0 / max(1.0, t_scale_s)

    r_new = cfg.alpha_new_area * (newly_treated_m2 * inv_a)
    r_repeat = -cfg.beta_repeat_area * (repeated_treated_m2 * inv_a)
    r_distance = -cfg.gamma_distance * (distance_traveled_m * inv_l)
    r_time = -cfg.eta_time * (dt_elapsed_s * inv_t)

    r_body_coll = -cfg.body_collision_penalty if body_collision else 0.0
    r_clr_breach = -cfg.clearance_breach_penalty if (clearance_breach and not body_collision) else 0.0
    r_filter = -cfg.safety_filter_penalty if safety_filter_intervened else 0.0
    r_obs_treat = -cfg.obstacle_treatment_penalty * (new_obstacle_intrusion_m2 * inv_a)

    # Continuous clearance warning shaping (dt-scaled so invariant to action frequency!)
    r_clr_shape = 0.0
    if cfg.clearance_shaping_weight > 0.0 and cfg.clearance_warning_margin_m > 1e-6:
        if clearance_margin_above_req_m < cfg.clearance_warning_margin_m:
            prox = max(0.0, min(1.0, 1.0 - clearance_margin_above_req_m / cfg.clearance_warning_margin_m))
            r_clr_shape = -cfg.clearance_shaping_weight * prox * (dt_elapsed_s * inv_t)

    r_stagnation = 0.0
    if cfg.stagnation_penalty_weight > 0.0 and time_since_new_coverage_s > cfg.stagnation_grace_s:
        r_stagnation = -cfg.stagnation_penalty_weight * (dt_elapsed_s * inv_t)

    r_safety = r_body_coll + r_clr_breach + r_filter + r_obs_treat + r_clr_shape
    r_success = cfg.success_bonus if coverage_completed_safely else 0.0

    total = float(r_new + r_repeat + r_distance + r_time + r_safety + r_stagnation + r_success)
    terms = {
        "new_area": float(r_new),
        "repeat": float(r_repeat),
        "distance": float(r_distance),
        "time": float(r_time),
        "body_collision": float(r_body_coll),
        "clearance_breach": float(r_clr_breach),
        "safety_filter": float(r_filter),
        "obstacle_treatment": float(r_obs_treat),
        "clearance_shaping": float(r_clr_shape),
        "stagnation": float(r_stagnation),
        "safety_total": float(r_safety),
        "success": float(r_success),
        "total": total,
    }
    return total, terms
