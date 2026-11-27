"""Continuous-curvature physically constrained agricultural vehicle dynamics for PPO.

Reuses the verified RK4 continuous-curvature vehicle model from rl_agricultural_env_v1
with strict sub-step constraint enforcement across every physics integration step:
  - 0 <= v <= v_allowed(kappa) <= v_max (forward-only, anticipatory braking)
  - |kappa| <= kappa_max
  - |d kappa / ds| <= sigma_max (and Ackermann steering-rate limit if configured)
  - Stationary steering lock (v == 0 => dtheta/dt = 0, dkappa/dt = 0)
  - -d_max <= a <= a_max, |omega| <= omega_max, |alpha| <= alpha_max

Action interfaces:
  - TARGET_SPEED_CURVATURE (Primary 2D): a = [a_v, a_kappa] in [-1, 1]^2
      v_target = ((a_v + 1) / 2) * v_max,  kappa_target = a_kappa * kappa_max
  - CURVATURE_ONLY (Optional 1D): a = [a_kappa] in [-1, 1]^1
      kappa_target = a_kappa * kappa_max,  v_target = v_allowed(kappa_target)
  - DIRECT_ACCEL_RATE (Auxiliary 2D): a = [a_accel, a_sigma] in [-1, 1]^2
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math

import numpy as np

from .robot_specs import RobotSpec


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


class ControlInterface(str, Enum):
    TARGET_SPEED_CURVATURE = "target_speed_curvature"  # Primary 2D: [a_v, a_kappa]
    CURVATURE_ONLY = "curvature_only"                  # Optional 1D: [a_kappa] with admissible speed policy
    DIRECT_ACCEL_RATE = "direct_accel_rate"            # Direct 2D: [a_req, u_sigma_req]

    @property
    def action_dim(self) -> int:
        return 1 if self == ControlInterface.CURVATURE_ONLY else 2


@dataclass
class VehicleState:
    x: float
    y: float
    theta: float
    v: float = 0.0
    kappa: float = 0.0
    t: float = 0.0
    s: float = 0.0
    energy_j: float = 0.0
    soc: float = 1.0

    @property
    def omega(self) -> float:
        return self.v * self.kappa

    def copy(self) -> VehicleState:
        return VehicleState(
            x=self.x, y=self.y, theta=self.theta, v=self.v, kappa=self.kappa,
            t=self.t, s=self.s, energy_j=self.energy_j, soc=self.soc,
        )

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta, self.v, self.kappa], dtype=np.float64)


@dataclass
class StepDiagnostics:
    """Complete requested vs executed control and inter-sample physical extrema record."""
    interface: str
    dt_total: float
    dt_phys: float
    n_substeps: int

    requested_cmd: tuple[float, float]
    executed_accel_m_s2: float
    executed_spatial_rate_inv_m2: float
    executed_temporal_rate_inv_m_s: float
    executed_v_end: float
    executed_kappa_end: float

    speed_saturated: bool = False
    turning_speed_limited: bool = False
    anticipatory_braking_active: bool = False
    curvature_saturated: bool = False
    curvature_rate_saturated: bool = False
    accel_saturated: bool = False
    angular_accel_saturated: bool = False
    stationary_steering_blocked: bool = False

    max_v_substep: float = 0.0
    min_v_substep: float = 0.0
    max_abs_kappa_substep: float = 0.0
    max_abs_omega_substep: float = 0.0
    max_abs_spatial_rate_substep: float = 0.0
    max_abs_accel_substep: float = 0.0
    max_abs_angular_accel_substep: float = 0.0
    max_turning_speed_excess_m_s: float = 0.0

    distance_traveled_m: float = 0.0
    energy_used_j: float = 0.0
    substep_states: list[tuple[float, float, float, float, float]] = field(default_factory=list)

    @property
    def any_saturation(self) -> bool:
        return (
            self.speed_saturated
            or self.turning_speed_limited
            or self.curvature_saturated
            or self.curvature_rate_saturated
            or self.accel_saturated
            or self.angular_accel_saturated
            or self.stationary_steering_blocked
        )


def _rk4_integrate_substep(
    x: float,
    y: float,
    theta: float,
    v: float,
    kappa: float,
    a: float,
    u_sigma: float,
    dt: float,
) -> tuple[float, float, float, float, float]:
    """4th-order Runge-Kutta integration of the continuous-curvature kinematic ODEs."""
    h2 = 0.5 * dt

    v0 = max(0.0, v)
    k1_0 = v0 * math.cos(theta)
    k1_1 = v0 * math.sin(theta)
    k1_2 = v0 * kappa
    k1_4 = v0 * u_sigma

    th1 = theta + h2 * k1_2
    v1 = max(0.0, v + h2 * a)
    kap1 = kappa + h2 * k1_4
    k2_0 = v1 * math.cos(th1)
    k2_1 = v1 * math.sin(th1)
    k2_2 = v1 * kap1
    k2_4 = v1 * u_sigma

    th2 = theta + h2 * k2_2
    kap2 = kappa + h2 * k2_4
    k3_0 = v1 * math.cos(th2)
    k3_1 = v1 * math.sin(th2)
    k3_2 = v1 * kap2
    k3_4 = v1 * u_sigma

    th3 = theta + dt * k3_2
    v3 = max(0.0, v + dt * a)
    kap3 = kappa + dt * k3_4
    k4_0 = v3 * math.cos(th3)
    k4_1 = v3 * math.sin(th3)
    k4_2 = v3 * kap3
    k4_4 = v3 * u_sigma

    c = dt / 6.0
    nx = x + c * (k1_0 + 2.0 * k2_0 + 2.0 * k3_0 + k4_0)
    ny = y + c * (k1_1 + 2.0 * k2_1 + 2.0 * k3_1 + k4_1)
    nth = wrap_angle(theta + c * (k1_2 + 2.0 * k2_2 + 2.0 * k3_2 + k4_2))
    nv = v3
    nkap = kappa + c * (k1_4 + 2.0 * k2_4 + 2.0 * k3_4 + k4_4)
    return nx, ny, nth, nv, nkap


class VehicleSimulator:
    """Physically constrained agricultural robot simulator supporting heterogeneous RobotSpecs."""

    def __init__(
        self,
        spec: RobotSpec,
        *,
        dt_phys: float = 0.02,
        interface: ControlInterface = ControlInterface.TARGET_SPEED_CURVATURE,
    ) -> None:
        if dt_phys <= 0.0 or dt_phys > 0.1:
            raise ValueError(f"dt_phys must be in (0, 0.1], got {dt_phys}")
        self.spec = spec
        self.dt_phys = float(dt_phys)
        self.interface = interface
        self.state = VehicleState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, spec.energy.initial_soc)

    def reset(
        self,
        x: float = 0.0,
        y: float = 0.0,
        theta: float = 0.0,
        v: float = 0.0,
        kappa: float = 0.0,
        soc: float | None = None,
    ) -> VehicleState:
        self.validate_state(v, kappa)
        init_soc = self.spec.energy.initial_soc if soc is None else float(soc)
        if not (0.0 <= init_soc <= 1.0):
            raise ValueError(f"soc must be in [0, 1], got {init_soc}")
        self.state = VehicleState(
            x=float(x),
            y=float(y),
            theta=wrap_angle(theta),
            v=float(v),
            kappa=float(kappa),
            t=0.0,
            s=0.0,
            energy_j=0.0,
            soc=init_soc,
        )
        return self.state.copy()

    def validate_state(self, v: float, kappa: float, tol: float = 1e-6) -> None:
        s = self.spec
        if not (math.isfinite(v) and math.isfinite(kappa)):
            raise ValueError("Non-finite velocity or curvature")
        if s.forward_only and v < -tol:
            raise ValueError(f"Negative speed v={v} violates forward_only constraint")
        if abs(kappa) > s.max_curvature_inv_m + tol:
            raise ValueError(f"|kappa|={abs(kappa):.5f} exceeds kappa_max={s.max_curvature_inv_m:.5f}")
        v_allow = s.allowed_speed(kappa)
        if v > v_allow + tol:
            raise ValueError(
                f"Initial speed v={v:.4f} exceeds turning-speed limit v_allowed(kappa={kappa:.4f})={v_allow:.4f}"
            )

    def map_rl_action(self, action_norm: np.ndarray | tuple[float, ...] | list[float]) -> tuple[float, float]:
        """Map normalized continuous RL action in [-1, 1]^d into physical target commands."""
        arr = np.asarray(action_norm, dtype=np.float64).ravel()
        s = self.spec
        if self.interface == ControlInterface.CURVATURE_ONLY:
            if arr.size < 1:
                raise ValueError("CURVATURE_ONLY requires at least 1 action element")
            ak = float(np.clip(arr[0], -1.0, 1.0))
            kappa_target = ak * s.max_curvature_inv_m
            v_target = s.allowed_speed(kappa_target)
            return v_target, kappa_target
        if arr.size < 2:
            raise ValueError(f"{self.interface.value} requires 2 action elements, got shape {arr.shape}")
        a0 = float(np.clip(arr[0], -1.0, 1.0))
        a1 = float(np.clip(arr[1], -1.0, 1.0))
        if self.interface == ControlInterface.TARGET_SPEED_CURVATURE:
            v_target = 0.5 * (a0 + 1.0) * s.max_straight_speed_m_s
            kappa_target = a1 * s.max_curvature_inv_m
            return v_target, kappa_target
        accel_req = a0 * s.max_linear_accel_m_s2 if a0 >= 0.0 else a0 * s.max_linear_decel_m_s2
        sigma_req = a1 * s.max_curvature_rate_inv_m2
        return accel_req, sigma_req

    def _compute_admissible_substep_controls(
        self,
        v: float,
        kappa: float,
        cmd_0: float,
        cmd_1: float,
        dt: float,
        flags: dict[str, bool],
    ) -> tuple[float, float]:
        s = self.spec
        kmax = s.max_curvature_inv_m
        a_max = s.max_linear_accel_m_s2
        d_max = s.max_linear_decel_m_s2
        is_target_mode = self.interface in (
            ControlInterface.TARGET_SPEED_CURVATURE,
            ControlInterface.CURVATURE_ONLY,
        )

        if is_target_mode:
            v_req_raw, k_req_raw = cmd_0, cmd_1
            if v_req_raw < 0.0 or v_req_raw > s.max_straight_speed_m_s + 1e-12:
                flags["speed_saturated"] = True
            if abs(k_req_raw) > kmax + 1e-12:
                flags["curvature_saturated"] = True

            v_req = min(max(v_req_raw, 0.0), s.max_straight_speed_m_s)
            k_req = min(max(k_req_raw, -kmax), kmax)

            v_cap_cur = s.allowed_speed(kappa)
            v_cap_tgt = s.allowed_speed(k_req)
            if v_req > v_cap_tgt + 1e-9:
                flags["turning_speed_limited"] = True
                if v > v_cap_tgt + 1e-6:
                    flags["anticipatory_braking_active"] = True
            v_goal = min(v_req, v_cap_cur, v_cap_tgt)

            a_des = (v_goal - v) / dt
            a_exec = min(max(a_des, -d_max), a_max)
            if abs(a_exec - a_des) > 1e-9:
                flags["accel_saturated"] = True
        else:
            a_req_raw = cmd_0
            a_exec = min(max(a_req_raw, -d_max), a_max)
            if abs(a_exec - a_req_raw) > 1e-9:
                flags["accel_saturated"] = True

        v_cap_now = s.allowed_speed(kappa)
        v_next_unclipped = v + a_exec * dt
        if v_next_unclipped < 0.0:
            a_exec = -v / dt
            flags["speed_saturated"] = True
        elif v_next_unclipped > v_cap_now:
            a_max_cap = max(-d_max, (v_cap_now - v) / dt)
            if a_exec > a_max_cap + 1e-12:
                a_exec = a_max_cap
                if v_cap_now < s.max_straight_speed_m_s - 1e-9:
                    flags["turning_speed_limited"] = True
                else:
                    flags["speed_saturated"] = True
        v_next = max(0.0, v + a_exec * dt)
        v_mid = 0.5 * (v + v_next)
        ds = v_mid * dt

        if ds <= 1e-12:
            if is_target_mode:
                if abs(cmd_1 - kappa) > 1e-6:
                    flags["stationary_steering_blocked"] = True
            else:
                if abs(cmd_1) > 1e-6:
                    flags["stationary_steering_blocked"] = True
            return a_exec, 0.0

        max_dk_dt = s.max_temporal_curvature_rate(v_mid, kappa)
        sigma_bound = min(s.max_curvature_rate_inv_m2, max_dk_dt / max(v_mid, 1e-12))

        if is_target_mode:
            k_req = min(max(cmd_1, -kmax), kmax)
            u_sigma_des = (k_req - kappa) / ds
            u_sigma = min(max(u_sigma_des, -sigma_bound), sigma_bound)
            if abs(u_sigma - u_sigma_des) > 1e-9:
                flags["curvature_rate_saturated"] = True
        else:
            u_sigma = min(max(cmd_1, -sigma_bound), sigma_bound)
            if abs(u_sigma - cmd_1) > 1e-9:
                flags["curvature_rate_saturated"] = True

        # Strict forward-invariant turning-speed envelope across both v_mid and v_next
        k_env = min(kmax, s.max_safe_curvature(v_next), s.max_safe_curvature(v_mid))
        k_next_prop = kappa + u_sigma * ds
        if abs(k_next_prop) > k_env + 1e-12 and abs(k_next_prop) > abs(kappa):
            k_target_safe = min(max(k_next_prop, -k_env), k_env)
            u_sigma = min(max((k_target_safe - kappa) / ds, -sigma_bound), sigma_bound)
            if k_env < kmax - 1e-9:
                flags["turning_speed_limited"] = True
            else:
                flags["curvature_saturated"] = True

        # Strict angular acceleration bound: |a*kappa + v^2 * u_sigma| <= alpha_max
        v_check = max(v, v_next)
        if v_check > 1e-9:
            alpha_exec = a_exec * kappa + (v_check * v_check) * u_sigma
            if abs(alpha_exec) > s.max_angular_accel_rad_s2 + 1e-12:
                lo_u = (-s.max_angular_accel_rad_s2 - a_exec * kappa) / (v_check * v_check)
                hi_u = (s.max_angular_accel_rad_s2 - a_exec * kappa) / (v_check * v_check)
                u_clamped = min(max(u_sigma, max(-sigma_bound, lo_u)), min(sigma_bound, hi_u))
                if abs(u_clamped - u_sigma) > 1e-9:
                    u_sigma = u_clamped
                    flags["angular_accel_saturated"] = True

        k_final = kappa + u_sigma * ds
        if abs(k_final) > kmax:
            k_final = min(max(k_final, -kmax), kmax)
            u_sigma = (k_final - kappa) / ds
            flags["curvature_saturated"] = True

        return a_exec, u_sigma

    def step(
        self,
        cmd: tuple[float, ...] | np.ndarray,
        dt: float,
        *,
        normalized_action: bool = False,
        record_substeps: bool = True,
    ) -> tuple[VehicleState, StepDiagnostics]:
        if dt <= 0.0 or not math.isfinite(dt):
            raise ValueError(f"dt must be positive and finite, got {dt}")
        if normalized_action:
            c0, c1 = self.map_rl_action(cmd)
        else:
            if self.interface == ControlInterface.CURVATURE_ONLY:
                arr = np.asarray(cmd, dtype=np.float64).ravel()
                c1 = float(arr[0])
                c0 = self.spec.allowed_speed(c1)
            else:
                c0, c1 = float(cmd[0]), float(cmd[1])
        if not (math.isfinite(c0) and math.isfinite(c1)):
            raise ValueError(f"Non-finite command {(c0, c1)}")

        n_sub = max(1, int(math.ceil(dt / self.dt_phys - 1e-12)))
        h = dt / n_sub

        flags = {
            "speed_saturated": False,
            "turning_speed_limited": False,
            "anticipatory_braking_active": False,
            "curvature_saturated": False,
            "curvature_rate_saturated": False,
            "accel_saturated": False,
            "angular_accel_saturated": False,
            "stationary_steering_blocked": False,
        }

        st = self.state
        x, y, th, v, kap = st.x, st.y, st.theta, st.v, st.kappa
        substates: list[tuple[float, float, float, float, float]] = [(x, y, th, v, kap)] if record_substeps else []

        max_v = v
        min_v = v
        max_k = abs(kap)
        max_w = abs(v * kap)
        max_sig = 0.0
        max_acc = 0.0
        max_ang_acc = 0.0
        max_v_excess = max(0.0, v - self.spec.allowed_speed(kap))

        dist_step = 0.0
        energy_step = 0.0
        sum_a = 0.0
        sum_sig = 0.0
        sum_dk_dt = 0.0

        energy_spec = self.spec.energy
        tool_active = self.spec.tool_always_active
        allow_fn = self.spec.allowed_speed

        for _ in range(n_sub):
            a_exec, u_sig = self._compute_admissible_substep_controls(v, kap, c0, c1, h, flags)
            nx, ny, nth, nv, nkap = _rk4_integrate_substep(x, y, th, v, kap, a_exec, u_sig, h)
            v_mid = 0.5 * (v + nv)
            k_mid = 0.5 * (kap + nkap)
            ds = v_mid * h

            dist_step += ds
            sum_a += a_exec * h
            sum_sig += u_sig * h
            sum_dk_dt += (v_mid * u_sig) * h

            ang_acc = abs(a_exec * kap + max(v, nv) * max(v, nv) * u_sig)
            if ang_acc > max_ang_acc:
                max_ang_acc = ang_acc

            if energy_spec.enabled:
                energy_step += energy_spec.power_watts(v_mid, k_mid, tool_active) * h

            x, y, th, v, kap = nx, ny, nth, nv, nkap

            if v > max_v:
                max_v = v
            if v < min_v:
                min_v = v
            ak = abs(kap)
            if ak > max_k:
                max_k = ak
            aw = abs(v * kap)
            if aw > max_w:
                max_w = aw
            if abs(u_sig) > max_sig:
                max_sig = abs(u_sig)
            if abs(a_exec) > max_acc:
                max_acc = abs(a_exec)
            v_exc = v - allow_fn(kap)
            if v_exc > max_v_excess:
                max_v_excess = v_exc

            if record_substeps:
                substates.append((x, y, th, v, kap))

        st.x = x
        st.y = y
        st.theta = th
        st.v = v
        st.kappa = kap
        st.t += dt
        st.s += dist_step
        st.energy_j += energy_step
        if energy_spec.enabled and energy_spec.battery_capacity_wh and energy_spec.battery_capacity_wh > 0.0:
            st.soc = max(0.0, st.soc - energy_step / (energy_spec.battery_capacity_wh * 3600.0))

        diag = StepDiagnostics(
            interface=self.interface.value,
            dt_total=dt,
            dt_phys=h,
            n_substeps=n_sub,
            requested_cmd=(c0, c1),
            executed_accel_m_s2=sum_a / dt,
            executed_spatial_rate_inv_m2=sum_sig / dt,
            executed_temporal_rate_inv_m_s=sum_dk_dt / dt,
            executed_v_end=v,
            executed_kappa_end=kap,
            speed_saturated=flags["speed_saturated"],
            turning_speed_limited=flags["turning_speed_limited"],
            anticipatory_braking_active=flags["anticipatory_braking_active"],
            curvature_saturated=flags["curvature_saturated"],
            curvature_rate_saturated=flags["curvature_rate_saturated"],
            accel_saturated=flags["accel_saturated"],
            angular_accel_saturated=flags["angular_accel_saturated"],
            stationary_steering_blocked=flags["stationary_steering_blocked"],
            max_v_substep=max_v,
            min_v_substep=min_v,
            max_abs_kappa_substep=max_k,
            max_abs_omega_substep=max_w,
            max_abs_spatial_rate_substep=max_sig,
            max_abs_accel_substep=max_acc,
            max_abs_angular_accel_substep=max_ang_acc,
            max_turning_speed_excess_m_s=max_v_excess,
            distance_traveled_m=dist_step,
            energy_used_j=energy_step,
            substep_states=substates,
        )
        return st.copy(), diag
