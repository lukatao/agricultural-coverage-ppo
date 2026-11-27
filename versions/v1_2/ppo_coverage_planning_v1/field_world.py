"""Portable agricultural field geometry (Families A-D), swept collision detection, and typed raycasting.

Repairs & upgrades over rl_agricultural_env_v1/field_collision.py:
  - Zero hardcoded local paths or coverage_benchmark imports (100% Kaggle-portable).
  - Separate exterior & internal-obstacle distance fields for continuous clearance sensing.
  - Safe chord decimation (<= 0.15 m) so long PPO action intervals (0.5-2.0 s) remain both
    fast and zero-tunneling accurate.
  - Typed ego-centric raycasting distinguishing exterior boundaries vs internal obstacles.
  - Explicit distinction between physical body collision and mandatory safety-clearance breach.
  - Artificial BCD interfaces remain purely optional analysis overlays (never policy input, never obstacles).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.prepared import prep

from .robot_specs import RobotSpec


class CollisionMode(str, Enum):
    TERMINATE = "terminate"              # Principal strict scientific baseline (body collision OR clearance breach terminates)
    TERMINATE_BODY_ONLY = "terminate_body_only"  # Curriculum option: body collision terminates; clearance breach penalized
    STOP_AT_PREFIX = "stop_at_prefix"    # Permissive safety-filter mode (explicitly flagged, not legal braking)
    REJECT_ACTION = "reject_action"      # Permissive safety-filter mode (explicitly flagged, zero motion)


@dataclass
class CollisionReport:
    """Detailed physical collision & safety-clearance audit for one simulation step."""
    safe: bool
    body_collision: bool = False
    clearance_breach: bool = False
    collision_type: str = "none"
    exterior_collision: bool = False
    obstacle_collision: bool = False
    exterior_clearance_breach: bool = False
    obstacle_clearance_breach: bool = False
    min_exterior_dist_m: float = math.inf
    min_obstacle_dist_m: float = math.inf
    min_body_clearance_m: float = math.inf
    safe_prefix_fraction: float = 1.0
    body_safe_prefix_fraction: float = 1.0
    crossed_artificial_boundary: bool = False
    artificial_crossings_count: int = 0
    implement_collision_status: str = "unverified"


def _ring_segments(ring_coords: Any) -> np.ndarray:
    pts = np.asarray(ring_coords, dtype=np.float64)[:, :2]
    if len(pts) < 2:
        return np.zeros((0, 4), dtype=np.float64)
    return np.hstack([pts[:-1], pts[1:]])


def _min_dist_segments_to_edges(path_xy: np.ndarray, edges: np.ndarray) -> float:
    """Exact minimum Euclidean distance between polyline (N, 2) and segment set (E, 4)."""
    if len(edges) == 0 or len(path_xy) == 0:
        return math.inf
    ax, ay, bx, by = edges[:, 0], edges[:, 1], edges[:, 2], edges[:, 3]
    abx, aby = bx - ax, by - ay
    ab2 = np.maximum(abx * abx + aby * aby, 1e-18)

    if len(path_xy) == 1:
        p = path_xy[0]
        apx, apy = p[0] - ax, p[1] - ay
        t = np.clip((apx * abx + apy * aby) / ab2, 0.0, 1.0)
        dx = apx - t * abx
        dy = apy - t * aby
        return float(np.sqrt(np.min(dx * dx + dy * dy)))

    best_d2 = math.inf
    for k in range(len(path_xy) - 1):
        cx, cy = float(path_xy[k, 0]), float(path_xy[k, 1])
        dx_, dy_ = float(path_xy[k + 1, 0]), float(path_xy[k + 1, 1])
        cdx, cdy = dx_ - cx, dy_ - cy
        cd2 = cdx * cdx + cdy * cdy

        c1 = abx * (cy - ay) - aby * (cx - ax)
        c2 = abx * (dy_ - ay) - aby * (dx_ - ax)
        c3 = cdx * (ay - cy) - cdy * (ax - cx)
        c4 = cdx * (by - cy) - cdy * (bx - cx)
        if np.any(((c1 * c2) < 0.0) & ((c3 * c4) < 0.0)):
            return 0.0

        t1 = np.clip(((cx - ax) * abx + (cy - ay) * aby) / ab2, 0.0, 1.0)
        m1 = float(np.min((cx - (ax + t1 * abx)) ** 2 + (cy - (ay + t1 * aby)) ** 2))
        if m1 < best_d2:
            best_d2 = m1

        t2 = np.clip(((dx_ - ax) * abx + (dy_ - ay) * aby) / ab2, 0.0, 1.0)
        m2 = float(np.min((dx_ - (ax + t2 * abx)) ** 2 + (dy_ - (ay + t2 * aby)) ** 2))
        if m2 < best_d2:
            best_d2 = m2

        if cd2 > 1e-18:
            s1 = np.clip(((ax - cx) * cdx + (ay - cy) * cdy) / cd2, 0.0, 1.0)
            m3 = float(np.min((ax - (cx + s1 * cdx)) ** 2 + (ay - (cy + s1 * cdy)) ** 2))
            if m3 < best_d2:
                best_d2 = m3

            s2 = np.clip(((bx - cx) * cdx + (by - cy) * cdy) / cd2, 0.0, 1.0)
            m4 = float(np.min((bx - (cx + s2 * cdx)) ** 2 + (by - (cy + s2 * cdy)) ** 2))
            if m4 < best_d2:
                best_d2 = m4

        if best_d2 <= 1e-18:
            return 0.0
    return math.sqrt(best_d2)


def _raycast_against_edges(
    x: float,
    y: float,
    rdx: np.ndarray,
    rdy: np.ndarray,
    edges: np.ndarray,
    max_range_m: float,
) -> np.ndarray:
    n_rays = len(rdx)
    if len(edges) == 0:
        return np.full(n_rays, max_range_m, dtype=np.float32)
    ax, ay = edges[:, 0], edges[:, 1]
    sx, sy = edges[:, 2] - ax, edges[:, 3] - ay
    qpx = ax - x
    qpy = ay - y

    denom = rdx[:, None] * sy[None, :] - rdy[:, None] * sx[None, :]
    num_t = qpx[None, :] * sy[None, :] - qpy[None, :] * sx[None, :]
    num_u = qpx[None, :] * rdy[:, None] - qpy[None, :] * rdx[:, None]

    valid = np.abs(denom) > 1e-12
    safe_denom = np.where(valid, denom, 1.0)
    t = num_t / safe_denom
    u = num_u / safe_denom
    hit = valid & (t >= 0.0) & (t <= max_range_m) & (u >= -1e-9) & (u <= 1.0 + 1e-9)
    t_masked = np.where(hit, t, max_range_m)
    return np.min(t_masked, axis=1).astype(np.float32)


class AgriculturalFieldWorld:
    """Fast portable geometric & physical world for Families A, B, C, D agricultural fields."""

    def __init__(
        self,
        polygon: Polygon,
        *,
        field_id: str = "custom_field",
        family: str = "A",
        bcd_cells: list[Polygon] | None = None,
        artificial_interfaces: list[LineString] | None = None,
        sdf_resolution_m: float = 0.5,
    ) -> None:
        if not isinstance(polygon, Polygon) or polygon.is_empty or not polygon.is_valid:
            raise ValueError("Field polygon must be a valid non-empty Shapely Polygon")

        self.field_id = str(field_id)
        self.family = str(family)
        self.polygon = polygon
        self.exterior_poly = Polygon(polygon.exterior)
        self.hole_polys = [Polygon(r) for r in polygon.interiors]
        self.holes_union = shapely.union_all(self.hole_polys) if self.hole_polys else Polygon()
        self.source_area_m2 = float(polygon.area)
        self.bounds = polygon.bounds
        self.width_m = float(self.bounds[2] - self.bounds[0])
        self.height_m = float(self.bounds[3] - self.bounds[1])
        self.center_x = 0.5 * (self.bounds[0] + self.bounds[2])
        self.center_y = 0.5 * (self.bounds[1] + self.bounds[3])
        self.bounding_diag_m = math.hypot(self.width_m, self.height_m)

        self._prep_poly = prep(polygon)
        self._prep_ext = prep(self.exterior_poly)

        self.ext_edges = _ring_segments(polygon.exterior.coords)
        hole_edge_list = [_ring_segments(r.coords) for r in polygon.interiors]
        self.obs_edges = np.vstack(hole_edge_list) if hole_edge_list else np.zeros((0, 4), dtype=np.float64)
        self.all_phys_edges = (
            np.vstack([self.ext_edges, self.obs_edges]) if len(self.obs_edges) else self.ext_edges
        )

        # Optional analysis overlays only (NEVER policy input, NEVER physical obstacles)
        self.bcd_cells = list(bcd_cells or [])
        self.artificial_interfaces = list(artificial_interfaces or [])
        art_edge_list = [_ring_segments(ls.coords) for ls in self.artificial_interfaces if not ls.is_empty]
        self.art_edges = np.vstack(art_edge_list) if art_edge_list else np.zeros((0, 4), dtype=np.float64)

        self._sdf_res = float(sdf_resolution_m)
        self._nav_cache: dict[tuple[float, float], Any] = {}
        self._build_sdfs()

    def _build_sdfs(self) -> None:
        xmin, ymin, xmax, ymax = self.bounds
        pad = 2.0 * self._sdf_res
        self._sdf_x0 = xmin - pad
        self._sdf_y0 = ymin - pad
        nx = max(4, int(math.ceil((xmax - xmin + 2.0 * pad) / self._sdf_res)) + 1)
        ny = max(4, int(math.ceil((ymax - ymin + 2.0 * pad) / self._sdf_res)) + 1)
        self._sdf_nx = nx
        self._sdf_ny = ny

        xs = self._sdf_x0 + (np.arange(nx, dtype=np.float64) + 0.5) * self._sdf_res
        ys = self._sdf_y0 + (np.arange(ny, dtype=np.float64) + 0.5) * self._sdf_res
        gx, gy = np.meshgrid(xs, ys)
        inside_ext = shapely.contains_xy(self.exterior_poly, gx, gy)
        if not self.holes_union.is_empty:
            in_holes = shapely.contains_xy(self.holes_union, gx, gy)
            inside = inside_ext & (~in_holes)
            obs_edt = distance_transform_edt(~in_holes) * self._sdf_res
        else:
            inside = inside_ext
            obs_edt = np.full((ny, nx), 1e6, dtype=np.float64)

        ext_edt = distance_transform_edt(inside_ext) * self._sdf_res
        tot_edt = distance_transform_edt(inside) * self._sdf_res

        self._sdf_total_edt = tot_edt.astype(np.float32)
        self._sdf_ext_edt = ext_edt.astype(np.float32)
        self._sdf_obs_edt = obs_edt.astype(np.float32)
        # Strict conservative lower bound for zero-miss broad-phase screening
        self._sdf_lb = np.maximum(0.0, tot_edt - 1.45 * self._sdf_res).astype(np.float32)

    def broadphase_clearance_lb(self, x: float, y: float) -> float:
        ix = int(math.floor((x - self._sdf_x0) / self._sdf_res))
        iy = int(math.floor((y - self._sdf_y0) / self._sdf_res))
        if 0 <= ix < self._sdf_nx and 0 <= iy < self._sdf_ny:
            return float(self._sdf_lb[iy, ix])
        return 0.0

    def query_exact_clearances(self, x: float, y: float) -> tuple[float, float]:
        """Exact center distances (d_exterior, d_obstacle) at point (x, y); 0 if outside/inside obstacle."""
        xy = np.array([[x, y]], dtype=np.float64)
        if not self._prep_ext.contains(Point(x, y)):
            d_ext = 0.0
        else:
            d_ext = _min_dist_segments_to_edges(xy, self.ext_edges)
        if len(self.obs_edges) == 0:
            d_obs = math.inf
        elif not self.holes_union.is_empty and shapely.contains_xy(self.holes_union, x, y):
            d_obs = 0.0
        else:
            d_obs = _min_dist_segments_to_edges(xy, self.obs_edges)
        return d_ext, d_obs

    def navigable_domain(self, spec: RobotSpec, extra_margin_m: float = 0.0) -> Any:
        key = (round(spec.exterior_center_clearance_m + extra_margin_m, 4),
               round(spec.obstacle_center_clearance_m + extra_margin_m, 4))
        if key in self._nav_cache:
            return self._nav_cache[key]
        shell = self.exterior_poly.buffer(-key[0], quad_segs=16)
        if shell.is_empty:
            res: Any = Polygon()
        elif not self.hole_polys:
            res = shell
        else:
            eroded_holes = [hp.buffer(key[1], quad_segs=16) for hp in self.hole_polys]
            res = shell.difference(shapely.union_all(eroded_holes))
        self._nav_cache[key] = res
        return res

    def is_point_valid_start(self, x: float, y: float, spec: RobotSpec, margin_m: float = 0.02) -> bool:
        if not self._prep_poly.contains(Point(x, y)):
            return False
        d_ext, d_obs = self.query_exact_clearances(x, y)
        if d_ext < spec.exterior_center_clearance_m + margin_m:
            return False
        if d_obs < spec.obstacle_center_clearance_m + margin_m:
            return False
        return True

    def sample_valid_start_pose(
        self,
        spec: RobotSpec,
        rng: np.random.Generator,
        *,
        margin_m: float = 0.35,
        max_attempts: int = 400,
    ) -> tuple[float, float, float]:
        """Sample a physically viable start pose (x, y, theta) in M with legal escape room.

        A forward-only continuous-curvature vehicle starting from rest (v=0, kappa=0) needs
        ~R_min + 1.6 m of forward/diagonal escape room to turn 90 deg without unavoidable
        wall contact. Rejecting start headings aimed straight into an immediately adjacent
        wall prevents physically inescapable Step-1 collisions while keeping full pose randomness.
        """
        nav = self.navigable_domain(spec, extra_margin_m=margin_m)
        if nav.is_empty:
            nav = self.navigable_domain(spec, extra_margin_m=0.05)
        if nav.is_empty:
            raise ValueError(f"Field {self.field_id} has empty navigable domain for robot {spec.robot_id}")
        nxmin, nymin, nxmax, nymax = nav.bounds
        pnav = prep(nav)
        min_ahead_m = spec.exterior_center_clearance_m + min(4.2, spec.min_turning_radius_m + 1.2)
        min_diag_m = spec.exterior_center_clearance_m + min(3.8, spec.min_turning_radius_m + 0.8)

        fallback_pose: tuple[float, float, float] | None = None
        for _ in range(max_attempts):
            rx = float(rng.uniform(nxmin, nxmax))
            ry = float(rng.uniform(nymin, nymax))
            if not pnav.contains(Point(rx, ry)):
                continue
            rth = float(rng.uniform(-math.pi, math.pi))
            if fallback_pose is None:
                fallback_pose = (rx, ry, rth)
            # Check forward and +-35 deg rays against physical walls to verify legal forward escape
            ang = np.array([rth, rth + 0.61, rth - 0.61], dtype=np.float64)
            rdx, rdy = np.cos(ang), np.sin(ang)
            d_ext = _raycast_against_edges(rx, ry, rdx, rdy, self.ext_edges, 20.0)
            d_obs = _raycast_against_edges(rx, ry, rdx, rdy, self.obs_edges, 20.0) if len(self.obs_edges) else np.full(3, 20.0)
            d_min = np.minimum(d_ext, d_obs)
            if d_min[0] >= min_ahead_m or (d_min[0] >= spec.exterior_center_clearance_m + 1.8 and max(d_min[1], d_min[2]) >= min_diag_m):
                return (rx, ry, rth)
        if fallback_pose is not None:
            return fallback_pose
        rp = nav.representative_point()
        return (float(rp.x), float(rp.y), float(rng.uniform(-math.pi, math.pi)))

    def _count_artificial_crossings(self, path_xy: np.ndarray) -> int:
        if len(self.art_edges) == 0 or len(path_xy) < 2:
            return 0
        ax, ay, bx, by = self.art_edges[:, 0], self.art_edges[:, 1], self.art_edges[:, 2], self.art_edges[:, 3]
        abx, aby = bx - ax, by - ay
        crossings = 0
        for k in range(len(path_xy) - 1):
            cx, cy = float(path_xy[k, 0]), float(path_xy[k, 1])
            dx, dy = float(path_xy[k + 1, 0]), float(path_xy[k + 1, 1])
            cdx, cdy = dx - cx, dy - cy
            c1 = abx * (cy - ay) - aby * (cx - ax)
            c2 = abx * (dy - ay) - aby * (dx - ax)
            c3 = cdx * (ay - cy) - cdy * (ax - cx)
            c4 = cdx * (by - cy) - cdy * (bx - cx)
            crossings += int(np.sum(((c1 * c2) <= 0.0) & ((c3 * c4) <= 0.0) & ((c1 != 0.0) | (c2 != 0.0))))
        return crossings

    def check_motion_substeps(
        self,
        substeps: list[tuple[float, float, float, float, float]] | np.ndarray,
        spec: RobotSpec,
    ) -> CollisionReport:
        """Continuous zero-tunneling collision & clearance check across full sub-step trajectory."""
        arr = np.asarray(substeps, dtype=np.float64)
        path_xy = arr[:, :2]
        if len(path_xy) == 0:
            return CollisionReport(safe=True, implement_collision_status=spec.implement_collision_status)

        art_hits = self._count_artificial_crossings(path_xy)
        r_body = spec.body_radius_m
        r_ext_req = spec.exterior_center_clearance_m
        r_obs_req = spec.obstacle_center_clearance_m
        r_req_max = max(r_ext_req, r_obs_req)

        p0x, p0y = float(path_xy[0, 0]), float(path_xy[0, 1])
        lb0 = self.broadphase_clearance_lb(p0x, p0y)
        if len(path_xy) >= 2:
            diffs = np.diff(path_xy, axis=0)
            step_path_len = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
        else:
            step_path_len = 0.0

        if lb0 > r_req_max + step_path_len + 0.05:
            # Query fast approximate exterior/obstacle EDT at end pose for continuous clearance diagnostics
            ex, ey = float(path_xy[-1, 0]), float(path_xy[-1, 1])
            ix = int(math.floor((ex - self._sdf_x0) / self._sdf_res))
            iy = int(math.floor((ey - self._sdf_y0) / self._sdf_res))
            if 0 <= ix < self._sdf_nx and 0 <= iy < self._sdf_ny:
                d_ext_est = max(lb0 - step_path_len, float(self._sdf_ext_edt[iy, ix]) - self._sdf_res)
                d_obs_est = (
                    max(lb0 - step_path_len, float(self._sdf_obs_edt[iy, ix]) - self._sdf_res)
                    if len(self.obs_edges)
                    else math.inf
                )
            else:
                d_ext_est = lb0 - step_path_len
                d_obs_est = (lb0 - step_path_len) if len(self.obs_edges) else math.inf
            return CollisionReport(
                safe=True,
                body_collision=False,
                clearance_breach=False,
                collision_type="none",
                min_exterior_dist_m=d_ext_est,
                min_obstacle_dist_m=d_obs_est,
                min_body_clearance_m=min(d_ext_est, d_obs_est) - r_body,
                safe_prefix_fraction=1.0,
                body_safe_prefix_fraction=1.0,
                crossed_artificial_boundary=(art_hits > 0),
                artificial_crossings_count=art_hits,
                implement_collision_status=spec.implement_collision_status,
            )

        # Safe chord decimation (<= 0.15 m chord => sagitta < 1.9 mm even at R_min = 1.5 m)
        if len(path_xy) > 6 and step_path_len > 1e-6:
            target_segs = max(4, int(math.ceil(step_path_len / 0.15)))
            stride = max(1, (len(path_xy) - 1) // target_segs)
            check_xy = path_xy[::stride]
            if not np.array_equal(check_xy[-1], path_xy[-1]):
                check_xy = np.vstack([check_xy, path_xy[-1:]])
        else:
            check_xy = path_xy

        d_ext = _min_dist_segments_to_edges(check_xy, self.ext_edges)
        d_obs = _min_dist_segments_to_edges(check_xy, self.obs_edges) if len(self.obs_edges) else math.inf

        out_ext = not bool(np.all(shapely.contains_xy(self.exterior_poly, check_xy[:, 0], check_xy[:, 1])))
        if out_ext:
            d_ext = 0.0
        in_obs = False
        if not self.holes_union.is_empty:
            if bool(np.any(shapely.contains_xy(self.holes_union, check_xy[:, 0], check_xy[:, 1]))):
                in_obs = True
                d_obs = 0.0

        ext_coll = out_ext or (d_ext < r_body - 1e-9)
        obs_coll = in_obs or (d_obs < r_body - 1e-9)
        ext_clr = (not ext_coll) and (d_ext < r_ext_req - 1e-9)
        obs_clr = (not obs_coll) and (d_obs < r_obs_req - 1e-9)

        body_coll = ext_coll or obs_coll
        clr_breach = ext_clr or obs_clr
        safe = not (body_coll or clr_breach)

        if obs_coll:
            ctype = "obstacle_collision"
        elif ext_coll:
            ctype = "exterior_collision"
        elif obs_clr:
            ctype = "obstacle_clearance_breach"
        elif ext_clr:
            ctype = "exterior_clearance_breach"
        else:
            ctype = "none"

        prefix_frac = 1.0
        body_prefix_frac = 1.0
        if not safe and len(path_xy) >= 2:
            prefix_frac = self._find_safe_prefix_fraction(path_xy, r_ext_req, r_obs_req)
        if body_coll and len(path_xy) >= 2:
            body_prefix_frac = self._find_safe_prefix_fraction(path_xy, r_body, r_body)

        min_d = min(d_ext, d_obs)
        return CollisionReport(
            safe=safe,
            body_collision=body_coll,
            clearance_breach=clr_breach,
            collision_type=ctype,
            exterior_collision=ext_coll,
            obstacle_collision=obs_coll,
            exterior_clearance_breach=ext_clr,
            obstacle_clearance_breach=obs_clr,
            min_exterior_dist_m=d_ext,
            min_obstacle_dist_m=d_obs,
            min_body_clearance_m=min_d - r_body,
            safe_prefix_fraction=prefix_frac,
            body_safe_prefix_fraction=body_prefix_frac,
            crossed_artificial_boundary=(art_hits > 0),
            artificial_crossings_count=art_hits,
            implement_collision_status=spec.implement_collision_status,
        )

    def _find_safe_prefix_fraction(self, path_xy: np.ndarray, r_ext: float, r_obs: float) -> float:
        n_seg = len(path_xy) - 1
        last_safe = 0
        for k in range(n_seg):
            seg = path_xy[k : k + 2]
            ok = self._prep_poly.contains(Point(float(seg[1, 0]), float(seg[1, 1])))
            if ok:
                if _min_dist_segments_to_edges(seg, self.ext_edges) < r_ext - 1e-9:
                    ok = False
                elif len(self.obs_edges) and _min_dist_segments_to_edges(seg, self.obs_edges) < r_obs - 1e-9:
                    ok = False
            if not ok:
                lo, hi = 0.0, 1.0
                p_a, p_b = seg[0], seg[1]
                for _ in range(10):
                    mid = 0.5 * (lo + hi)
                    pm = p_a + mid * (p_b - p_a)
                    sub = np.vstack([p_a, pm])
                    ok_m = self._prep_poly.contains(Point(float(pm[0]), float(pm[1])))
                    if ok_m:
                        if _min_dist_segments_to_edges(sub, self.ext_edges) < r_ext - 1e-9:
                            ok_m = False
                        elif len(self.obs_edges) and _min_dist_segments_to_edges(sub, self.obs_edges) < r_obs - 1e-9:
                            ok_m = False
                    if ok_m:
                        lo = mid
                    else:
                        hi = mid
                return (k + lo) / n_seg
            last_safe = k + 1
        return last_safe / n_seg

    def cast_typed_lidar_rays(
        self,
        x: float,
        y: float,
        theta: float,
        *,
        n_rays: int = 24,
        max_range_m: float = 36.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cast ego-centric rays returning SEPARATE normalized distances to exterior and internal obstacles.

        Returns:
            ``(ext_dist_m, obs_dist_m)`` each shape ``(n_rays,)`` in meters clipped to ``[0, max_range_m]``.
            Ray 0 points straight ahead along ``theta``; rays span ``[0, 2*pi)`` CCW.
            Artificial BCD boundaries are never visible as obstacles.
        """
        angles = theta + np.linspace(0.0, 2.0 * math.pi, n_rays, endpoint=False)
        rdx = np.cos(angles)
        rdy = np.sin(angles)
        d_ext = _raycast_against_edges(x, y, rdx, rdy, self.ext_edges, max_range_m)
        if len(self.obs_edges) > 0:
            d_obs = _raycast_against_edges(x, y, rdx, rdy, self.obs_edges, max_range_m)
        else:
            d_obs = np.full(n_rays, max_range_m, dtype=np.float32)
        return d_ext, d_obs
