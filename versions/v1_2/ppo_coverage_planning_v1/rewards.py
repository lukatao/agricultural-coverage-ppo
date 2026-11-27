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

    # Continuous low-speed / stationary idle penalty (zero when v >= idle_speed_threshold_frac * v_max)
    # Prevents the risk-averse v=0 parking local optimum without distorting legal turning speeds (0.5 m/s > 0.2*1.5 m/s).
    idle_penalty_weight: float = 0.0
    idle_speed_threshold_frac: float = 0.20

    # Optional stagnation penalty after prolonged time without meaningful new coverage
    stagnation_penalty_weight: float = 0.0
    stagnation_grace_s: float = 15.0

    # v1.2 Unproductive circling / turning penalty on already-treated land (dt-scaled, 0 on straight or fresh land)
    # Eliminates the CURVATURE_ONLY tight-orbit speed-reduction exploit where turning tight slows speed to cut repeat cost.
    unproductive_turn_weight: float = 0.0

    # v1.2 One-time progressive safe-coverage milestone bonuses: ((threshold_pct, bonus), ...)
    # Each fires at most once per episode when safely crossing threshold_pct (strictly telescoping & dt-invariant).
    coverage_milestones: tuple[tuple[float, float], ...] = ()

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
    if name == "v1_1_low_repeat":
        # Ablation 1: isolate reducing repeated-treatment & distance penalties for cheaper transit
        return RewardConfig(
            preset_name="v1_1_low_repeat",
            alpha_new_area=100.0,
            beta_repeat_area=3.0,
            gamma_distance=2.0,
            eta_time=5.0,
            body_collision_penalty=40.0,
            clearance_breach_penalty=35.0,
            safety_filter_penalty=2.0,
            success_bonus=30.0,
        )
    if name == "v1_1_anti_idle":
        # Ablation 2 (Primary v1.1 candidate): full-speed treated transit (-10/T) & mid-speed transit (-7.5/T)
        # are 1.5x-2.0x cheaper than sitting frozen (-15/T); legal min-radius turns (v >= 0.28*v_max across all
        # 4 robots) incur ZERO idle penalty; collision (-60/-65) strictly worse than full 3.5*T timeout (-52.5);
        # stronger completion bonus (+60).
        return RewardConfig(
            preset_name="v1_1_anti_idle",
            alpha_new_area=100.0,
            beta_repeat_area=3.0,
            gamma_distance=2.0,
            eta_time=5.0,
            idle_penalty_weight=10.0,
            idle_speed_threshold_frac=0.25,
            body_collision_penalty=65.0,
            clearance_breach_penalty=60.0,
            safety_filter_penalty=2.0,
            success_bonus=60.0,
        )
    if name == "v1_1_anti_idle_stag":
        # Ablation 3: v1_1_anti_idle + explicit stagnation penalty after 15 s without meaningful coverage
        return RewardConfig(
            preset_name="v1_1_anti_idle_stag",
            alpha_new_area=100.0,
            beta_repeat_area=3.0,
            gamma_distance=2.0,
            eta_time=5.0,
            idle_penalty_weight=10.0,
            idle_speed_threshold_frac=0.25,
            stagnation_penalty_weight=4.0,
            stagnation_grace_s=15.0,
            body_collision_penalty=80.0,
            clearance_breach_penalty=75.0,
            safety_filter_penalty=2.0,
            success_bonus=60.0,
        )
    if name == "v1_1_balanced":
        # Empirically refined v1.1 candidate after Batch 1 ablations: keeps moderate collision penalties
        # (-40/-45, avoiding wall-terror paralysis seen at -60/-80) while combining low repeat/distance
        # cost, gentle anti-idle (-3/T), gentle stagnation shaping (-2/T after 15s), and +60 completion.
        return RewardConfig(
            preset_name="v1_1_balanced",
            alpha_new_area=100.0,
            beta_repeat_area=3.0,
            gamma_distance=2.0,
            eta_time=5.0,
            idle_penalty_weight=3.0,
            idle_speed_threshold_frac=0.25,
            stagnation_penalty_weight=2.0,
            stagnation_grace_s=15.0,
            body_collision_penalty=45.0,
            clearance_breach_penalty=40.0,
            safety_filter_penalty=2.0,
            success_bonus=60.0,
        )
    if name == "v1_2_anti_circle":
        # v1.2 Ablation A: closes CURVATURE_ONLY tight-orbit speed loophole so straight transit (-9.0/T)
        # is strictly cheaper than moderate (-9.3/T to -10.7/T) or tight (-11.5/T) circling over treated land.
        # Worst-case 3.5*T timeout (-40.25) < clearance breach (-42) / body collision (-47).
        return RewardConfig(
            preset_name="v1_2_anti_circle",
            alpha_new_area=100.0,
            beta_repeat_area=5.0,
            gamma_distance=1.0,
            eta_time=3.0,
            unproductive_turn_weight=6.5,
            body_collision_penalty=52.0,
            clearance_breach_penalty=46.0,
            safety_filter_penalty=2.0,
            success_bonus=30.0,
        )
    if name == "v1_2_anti_circle_milestones":
        # v1.2 Ablation B: v1_2_anti_circle + one-time safe progressive coverage milestones (85%..99%)
        # giving a smooth, telescoping reward gradient to finish outer edges/corners & terminate early.
        return RewardConfig(
            preset_name="v1_2_anti_circle_milestones",
            alpha_new_area=100.0,
            beta_repeat_area=5.0,
            gamma_distance=1.0,
            eta_time=3.0,
            unproductive_turn_weight=6.5,
            coverage_milestones=(
                (85.0, 5.0),
                (90.0, 10.0),
                (93.0, 15.0),
                (95.0, 25.0),
                (97.0, 35.0),
                (99.0, 50.0),
            ),
            body_collision_penalty=52.0,
            clearance_breach_penalty=46.0,
            safety_filter_penalty=2.0,
            success_bonus=50.0,
        )
    if name == "v1_2_strict_anti_circle":
        # v1.2 Ablation C: stronger overlap push (beta=7.0) + stronger anti-circle (w_turn=8.0) + milestones.
        # Straight transit (-11.5/T) < tight circle (-13.83/T); 3.5*T worst-case (-48.4) < breach (-56)/coll (-62).
        return RewardConfig(
            preset_name="v1_2_strict_anti_circle",
            alpha_new_area=100.0,
            beta_repeat_area=7.0,
            gamma_distance=1.5,
            eta_time=3.0,
            unproductive_turn_weight=8.0,
            coverage_milestones=(
                (85.0, 5.0),
                (90.0, 10.0),
                (93.0, 15.0),
                (95.0, 25.0),
                (97.0, 35.0),
                (99.0, 50.0),
            ),
            body_collision_penalty=62.0,
            clearance_breach_penalty=56.0,
            safety_filter_penalty=2.0,
            success_bonus=50.0,
        )
    raise KeyError(
        f"Unknown reward preset {name!r}; expected one of "
        "('simple_v1', 'shaped_v1', 'v1_1_low_repeat', 'v1_1_anti_idle', 'v1_1_anti_idle_stag', 'v1_1_balanced', "
        "'v1_2_anti_circle', 'v1_2_anti_circle_milestones', 'v1_2_strict_anti_circle')"
    )


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
    executed_speed_frac: float = 1.0,
    stagnation_truncated_remaining_s: float = 0.0,
    abs_kappa_frac: float = 0.0,
    coverage_pct_before: float = 0.0,
    coverage_pct_after: float = 0.0,
    safe_clean_so_far: bool = True,
) -> tuple[float, dict[str, float]]:
    """Compute scalar reward and exact breakdown dict for one executed step."""
    inv_a = 1.0 / max(1.0, field_area_m2)
    inv_l = 1.0 / max(1.0, l_scale_m)
    inv_t = 1.0 / max(1.0, t_scale_s)

    r_new = cfg.alpha_new_area * (newly_treated_m2 * inv_a)
    r_repeat = -cfg.beta_repeat_area * (repeated_treated_m2 * inv_a)
    r_distance = -cfg.gamma_distance * (distance_traveled_m * inv_l)
    r_time = -cfg.eta_time * (dt_elapsed_s * inv_t)

    r_idle = 0.0
    if cfg.idle_penalty_weight > 0.0 and cfg.idle_speed_threshold_frac > 1e-6:
        if executed_speed_frac < cfg.idle_speed_threshold_frac:
            deficit = max(0.0, min(1.0, 1.0 - executed_speed_frac / cfg.idle_speed_threshold_frac))
            r_idle = -cfg.idle_penalty_weight * deficit * (dt_elapsed_s * inv_t)

    # Unproductive circling penalty: 0 when straight (abs_kappa_frac=0) or when sweeping fresh land (unprod=0)
    r_circle = 0.0
    if cfg.unproductive_turn_weight > 0.0 and abs_kappa_frac > 1e-6:
        swept_total = newly_treated_m2 + repeated_treated_m2
        unproductive_frac = (1.0 - newly_treated_m2 / swept_total) if swept_total > 1e-6 else 1.0
        unproductive_frac = max(0.0, min(1.0, unproductive_frac))
        r_circle = -cfg.unproductive_turn_weight * unproductive_frac * max(0.0, min(1.0, abs_kappa_frac)) * (dt_elapsed_s * inv_t)

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
    # Exploit-proof stagnation truncation penalty: charges remaining idle/orbit budget so quitting early never pays!
    if stagnation_truncated_remaining_s > 0.0:
        r_stagnation -= (
            cfg.eta_time + cfg.idle_penalty_weight + cfg.stagnation_penalty_weight + cfg.unproductive_turn_weight
        ) * (stagnation_truncated_remaining_s * inv_t)

    r_safety = r_body_coll + r_clr_breach + r_filter + r_obs_treat + r_clr_shape
    r_milestone = 0.0
    if cfg.coverage_milestones and safe_clean_so_far and coverage_pct_after > coverage_pct_before:
        for thresh_pct, bonus in cfg.coverage_milestones:
            if coverage_pct_before < thresh_pct <= coverage_pct_after:
                r_milestone += float(bonus)
    r_success = cfg.success_bonus if coverage_completed_safely else 0.0

    total = float(r_new + r_repeat + r_distance + r_time + r_idle + r_circle + r_safety + r_stagnation + r_milestone + r_success)
    terms = {
        "new_area": float(r_new),
        "repeat": float(r_repeat),
        "distance": float(r_distance),
        "time": float(r_time),
        "idle": float(r_idle),
        "circle": float(r_circle),
        "body_collision": float(r_body_coll),
        "clearance_breach": float(r_clr_breach),
        "safety_filter": float(r_filter),
        "obstacle_treatment": float(r_obs_treat),
        "clearance_shaping": float(r_clr_shape),
        "stagnation": float(r_stagnation),
        "safety_total": float(r_safety),
        "milestone": float(r_milestone),
        "success": float(r_success),
        "total": total,
    }
    return total, terms
