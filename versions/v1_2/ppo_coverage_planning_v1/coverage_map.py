"""Global & local agricultural coverage accounting for PPO training and exact evaluation.

Repairs over rl_agricultural_env_v1/coverage_map.py:
  1. Exact non-bloated swept-quad rasterization (tiny movements cannot farm coverage cells).
  2. Step-invariant telescoping repeated-treatment delta:
         R_cum(t) = max(0, G_t - C_t),  delta_R(t) = R_cum(t) - R_cum(t-1)
     eliminating per-step raster quantization drift and guaranteeing invariance across
     PPO action intervals (0.2 s, 0.5 s, 1.0 s, 2.0 s) and integration sub-steps.
  3. Largest connected untreated region monitoring (scipy.ndimage.label).
  4. Upgraded 5-channel ego-aligned local map (default 24x24 @ 36 m = 1.5 m/cell) including
     permissible robot-center navigable space M for the active RobotSpec.
  5. Aspect-preserving 6-channel global map (default 32x32, supporting both ego_aligned and
     world frames) using 2x2 sub-cell aggregation so thin obstacles and narrow corridors are
     preserved without geometric distortion.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import binary_dilation, label as ndimage_label
import shapely
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry


@dataclass
class CoverageStepDelta:
    """Incremental treatment metrics produced by one simulation step."""
    newly_treated_m2: float
    repeated_treated_m2: float
    gross_in_field_m2: float
    new_outside_field_m2: float
    new_obstacle_intrusion_m2: float
    coverage_pct_after: float
    untreated_m2_after: float


@dataclass
class CoverageSnapshot:
    """Global cumulative treatment metrics over original field domain F."""
    source_area_m2: float
    raster_field_area_m2: float
    coverage_pct: float
    covered_area_m2: float
    untreated_area_m2: float
    gross_treated_in_field_m2: float
    repeated_treatment_m2: float
    repeated_treatment_pct: float
    outside_field_treatment_m2: float
    obstacle_treatment_m2: float
    resolution_m: float
    memory_bytes: int


def _areal(geom: BaseGeometry) -> BaseGeometry:
    if geom.is_empty:
        return Polygon()
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    parts = [p for p in shapely.get_parts(geom) if p.geom_type in ("Polygon", "MultiPolygon") and p.area > 0.0]
    return shapely.union_all(parts) if parts else Polygon()


class IncrementalCoverageMap:
    """Fast incremental raster coverage tracker + local/global observation map extractor."""

    def __init__(
        self,
        field_polygon: Polygon,
        *,
        resolution_m: float = 0.50,
        pad_m: float = 6.0,
        navigable_polygon: BaseGeometry | None = None,
    ) -> None:
        if resolution_m <= 0.0:
            raise ValueError(f"resolution_m must be positive, got {resolution_m}")
        self.polygon = field_polygon
        self.source_area_m2 = float(field_polygon.area)
        self.resolution_m = float(resolution_m)
        self.cell_area_m2 = self.resolution_m * self.resolution_m

        xmin, ymin, xmax, ymax = field_polygon.bounds
        self.field_bounds = (float(xmin), float(ymin), float(xmax), float(ymax))
        self.field_cx = 0.5 * (xmin + xmax)
        self.field_cy = 0.5 * (ymin + ymax)
        self.field_width_m = float(xmax - xmin)
        self.field_height_m = float(ymax - ymin)
        self.field_diag_m = math.hypot(self.field_width_m, self.field_height_m)

        self.x0 = xmin - pad_m
        self.y0 = ymin - pad_m
        self.nx = max(4, int(math.ceil((xmax - xmin + 2.0 * pad_m) / self.resolution_m)))
        self.ny = max(4, int(math.ceil((ymax - ymin + 2.0 * pad_m) / self.resolution_m)))

        self.xs = self.x0 + (np.arange(self.nx, dtype=np.float64) + 0.5) * self.resolution_m
        self.ys = self.y0 + (np.arange(self.ny, dtype=np.float64) + 0.5) * self.resolution_m
        gx, gy = np.meshgrid(self.xs, self.ys)

        self.in_field = shapely.contains_xy(field_polygon, gx, gy)
        hole_polys = [Polygon(r) for r in field_polygon.interiors]
        if hole_polys:
            holes_u = shapely.union_all(hole_polys)
            self.in_obstacle = shapely.contains_xy(holes_u, gx, gy)
            self.in_obstacle_dilated = binary_dilation(self.in_obstacle, iterations=2)
        else:
            self.in_obstacle = np.zeros((self.ny, self.nx), dtype=bool)
            self.in_obstacle_dilated = self.in_obstacle
        self.outside_ext = (~self.in_field) & (~self.in_obstacle)

        if navigable_polygon is not None and not navigable_polygon.is_empty:
            self.in_navigable = shapely.contains_xy(navigable_polygon, gx, gy)
        else:
            self.in_navigable = self.in_field.copy()

        self.field_cell_count = int(np.sum(self.in_field))
        self.raster_field_area_m2 = self.field_cell_count * self.cell_area_m2
        self._area_scale = (self.source_area_m2 / self.raster_field_area_m2) if self.raster_field_area_m2 > 0 else 1.0

        self.covered = np.zeros((self.ny, self.nx), dtype=bool)
        self.treated_count = np.zeros((self.ny, self.nx), dtype=np.uint16)
        self._covered_field_cells = 0
        self._covered_obs_cells = 0
        self._covered_out_cells = 0
        self._gross_in_field_m2 = 0.0
        self._cum_repeated_m2 = 0.0

        # Caches for fast observation map extraction
        self._local_grid_cache: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}
        self._global_grid_cache: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, float]] = {}

        # Precompute field cell coordinates (and 2x strided coarse subset) for fast ego-centric global untreated cues
        fy, fx = np.nonzero(self.in_field)
        self._field_ir = fy.astype(np.int32)
        self._field_ic = fx.astype(np.int32)
        self._field_wx = self.xs[self._field_ic].astype(np.float32)
        self._field_wy = self.ys[self._field_ir].astype(np.float32)
        coarse_mask = (self._field_ir % 2 == 0) & (self._field_ic % 2 == 0)
        if int(np.sum(coarse_mask)) >= 64:
            self._coarse_ir = self._field_ir[coarse_mask]
            self._coarse_ic = self._field_ic[coarse_mask]
            self._coarse_wx = self._field_wx[coarse_mask]
            self._coarse_wy = self._field_wy[coarse_mask]
        else:
            self._coarse_ir = self._field_ir
            self._coarse_ic = self._field_ic
            self._coarse_wx = self._field_wx
            self._coarse_wy = self._field_wy

    def set_navigable_polygon(self, navigable_polygon: BaseGeometry | None) -> None:
        gx, gy = np.meshgrid(self.xs, self.ys)
        if navigable_polygon is not None and not navigable_polygon.is_empty:
            self.in_navigable = shapely.contains_xy(navigable_polygon, gx, gy)
        else:
            self.in_navigable = self.in_field.copy()

    def reset(self) -> None:
        self.covered.fill(False)
        self.treated_count.fill(0)
        self._covered_field_cells = 0
        self._covered_obs_cells = 0
        self._covered_out_cells = 0
        self._gross_in_field_m2 = 0.0
        self._cum_repeated_m2 = 0.0

    def apply_initial_coverage_mask(self, mask: np.ndarray) -> None:
        """Optionally initialize partial coverage for explicitly labeled experiments."""
        m = np.asarray(mask, dtype=bool)
        if m.shape != self.covered.shape:
            raise ValueError(f"Expected mask shape {self.covered.shape}, got {m.shape}")
        self.covered = m & self.in_field
        self.treated_count = self.covered.astype(np.uint16)
        self._covered_field_cells = int(np.sum(self.covered))
        self._covered_obs_cells = 0
        self._covered_out_cells = 0
        self._gross_in_field_m2 = self._covered_field_cells * self.cell_area_m2 * self._area_scale
        self._cum_repeated_m2 = 0.0

    @property
    def memory_bytes(self) -> int:
        return int(
            self.in_field.nbytes
            + self.in_obstacle.nbytes
            + self.outside_ext.nbytes
            + self.in_navigable.nbytes
            + self.covered.nbytes
            + self.treated_count.nbytes
            + self.xs.nbytes
            + self.ys.nbytes
        )

    def _rasterize_swept_quad(
        self,
        p0x: float,
        p0y: float,
        th0: float,
        p1x: float,
        p1y: float,
        th1: float,
        half_w: float,
    ) -> tuple[int, int, int, int, np.ndarray] | None:
        """Exact non-bloated point-in-quad rasterization of the swept implement bar.

        Uses exact oriented triangle tests on the quad [p0-hw*n0, p0+hw*n0, p1+hw*n1, p1-hw*n1]
        with ZERO artificial forward/backward thickness inflation so tiny movements cannot
        prematurely claim cells outside the physically swept area.
        """
        n0x, n0y = -math.sin(th0), math.cos(th0)
        n1x, n1y = -math.sin(th1), math.cos(th1)

        ax, ay = p0x - half_w * n0x, p0y - half_w * n0y
        bx, by = p0x + half_w * n0x, p0y + half_w * n0y
        cx, cy = p1x + half_w * n1x, p1y + half_w * n1y
        dx, dy = p1x - half_w * n1x, p1y - half_w * n1y

        qxmin = min(ax, bx, cx, dx)
        qxmax = max(ax, bx, cx, dx)
        qymin = min(ay, by, cy, dy)
        qymax = max(ay, by, cy, dy)

        inv_res = 1.0 / self.resolution_m
        c0 = max(0, int(math.floor((qxmin - self.x0) * inv_res)))
        c1 = min(self.nx, int(math.ceil((qxmax - self.x0) * inv_res)) + 1)
        r0 = max(0, int(math.floor((qymin - self.y0) * inv_res)))
        r1 = min(self.ny, int(math.ceil((qymax - self.y0) * inv_res)) + 1)
        if c0 >= c1 or r0 >= r1:
            return None

        lx = self.xs[c0:c1][None, :]
        ly = self.ys[r0:r1][:, None]

        def _in_tri(vx0: float, vy0: float, vx1: float, vy1: float, vx2: float, vy2: float) -> np.ndarray:
            e0 = (vx1 - vx0) * (ly - vy0) - (vy1 - vy0) * (lx - vx0)
            e1 = (vx2 - vx1) * (ly - vy1) - (vy2 - vy1) * (lx - vx1)
            e2 = (vx0 - vx2) * (ly - vy2) - (vy0 - vy2) * (lx - vx2)
            pos = (e0 >= -1e-9) & (e1 >= -1e-9) & (e2 >= -1e-9)
            neg = (e0 <= 1e-9) & (e1 <= 1e-9) & (e2 <= 1e-9)
            return pos | neg

        hit = _in_tri(ax, ay, bx, by, cx, cy) | _in_tri(ax, ay, cx, cy, dx, dy)
        return r0, r1, c0, c1, hit

    def _bar_in_field_fraction(self, px: float, py: float, th: float, half_w: float, n_samples: int = 11) -> float:
        """Exact continuous fraction of the implement bar at (px, py, th) lying inside F."""
        nx_v, ny_v = -math.sin(th), math.cos(th)
        offsets = np.linspace(-half_w, half_w, n_samples)
        bx = px + offsets * nx_v
        by = py + offsets * ny_v
        inv_res = 1.0 / self.resolution_m
        ic = np.floor((bx - self.x0) * inv_res).astype(np.int32)
        ir = np.floor((by - self.y0) * inv_res).astype(np.int32)
        valid = (ic >= 0) & (ic < self.nx) & (ir >= 0) & (ir < self.ny)
        if not np.any(valid):
            return 0.0
        ic_c = np.clip(ic, 0, self.nx - 1)
        ir_c = np.clip(ir, 0, self.ny - 1)
        in_f = valid & self.in_field[ir_c, ic_c]
        return float(np.mean(in_f))

    def update_from_substeps(
        self,
        substeps: list[tuple[float, float, float, float, float]] | np.ndarray,
        working_width_m: float,
        *,
        tool_active: bool = True,
    ) -> CoverageStepDelta:
        """Update global coverage incrementally along all physics sub-steps of one step."""
        if not tool_active or len(substeps) < 2:
            snap = self.snapshot()
            return CoverageStepDelta(0.0, 0.0, 0.0, 0.0, 0.0, snap.coverage_pct, snap.untreated_area_m2)

        arr = np.asarray(substeps, dtype=np.float64)
        half_w = 0.5 * float(working_width_m)

        diffs = np.diff(arr[:, :2], axis=0)
        ds_arr = np.hypot(diffs[:, 0], diffs[:, 1])
        total_ds = float(np.sum(ds_arr))
        if total_ds <= 1e-12:
            snap = self.snapshot()
            return CoverageStepDelta(0.0, 0.0, 0.0, 0.0, 0.0, snap.coverage_pct, snap.untreated_area_m2)

        # Decimate dense sub-steps to ~0.25 m chords for fast, well-conditioned quads (always keeping start & end)
        key_idx = [0]
        acc = 0.0
        for k in range(len(ds_arr)):
            acc += float(ds_arr[k])
            if acc >= 0.25 or k == len(ds_arr) - 1:
                key_idx.append(k + 1)
                acc = 0.0
        if key_idx[-1] != len(arr) - 1:
            key_idx.append(len(arr) - 1)

        pad = half_w + self.resolution_m
        sxmin = float(np.min(arr[:, 0])) - pad
        sxmax = float(np.max(arr[:, 0])) + pad
        symin = float(np.min(arr[:, 1])) - pad
        symax = float(np.max(arr[:, 1])) + pad

        inv_res = 1.0 / self.resolution_m
        C0 = max(0, int(math.floor((sxmin - self.x0) * inv_res)))
        C1 = min(self.nx, int(math.ceil((sxmax - self.x0) * inv_res)) + 1)
        R0 = max(0, int(math.floor((symin - self.y0) * inv_res)))
        R1 = min(self.ny, int(math.ceil((symax - self.y0) * inv_res)) + 1)
        if C0 >= C1 or R0 >= R1:
            snap = self.snapshot()
            return CoverageStepDelta(0.0, 0.0, 0.0, 0.0, 0.0, snap.coverage_pct, snap.untreated_area_m2)

        step_mask = np.zeros((R1 - R0, C1 - C0), dtype=bool)
        gross_in_field_step = 0.0

        for j in range(len(key_idx) - 1):
            i0, i1 = key_idx[j], key_idx[j + 1]
            seg_ds = float(np.sum(ds_arr[i0:i1]))
            if seg_ds <= 1e-12:
                continue
            p0x, p0y, th0 = float(arr[i0, 0]), float(arr[i0, 1]), float(arr[i0, 2])
            p1x, p1y, th1 = float(arr[i1, 0]), float(arr[i1, 1]), float(arr[i1, 2])
            res_q = self._rasterize_swept_quad(p0x, p0y, th0, p1x, p1y, th1, half_w)
            if res_q is not None:
                r0, r1, c0, c1, qmask = res_q
                step_mask[r0 - R0 : r1 - R0, c0 - C0 : c1 - C0] |= qmask

            # Continuous gross swept area inside F via midpoint implement-bar sampling (exact even when seg_ds < res!)
            mx, my = 0.5 * (p0x + p1x), 0.5 * (p0y + p1y)
            dth = (th1 - th0 + math.pi) % (2.0 * math.pi) - math.pi
            th_m = th0 + 0.5 * dth
            in_frac = self._bar_in_field_fraction(mx, my, th_m, half_w)
            gross_in_field_step += (working_width_m * seg_ds) * in_frac

        cov_view = self.covered[R0:R1, C0:C1]
        in_f_view = self.in_field[R0:R1, C0:C1]
        in_o_view = self.in_obstacle[R0:R1, C0:C1]
        out_view = self.outside_ext[R0:R1, C0:C1]

        newly_cov = step_mask & (~cov_view)
        new_f_cells = int(np.sum(newly_cov & in_f_view))
        new_o_cells = int(np.sum(newly_cov & in_o_view))
        new_out_cells = int(np.sum(newly_cov & out_view))

        cov_view |= step_mask
        self.treated_count[R0:R1, C0:C1] += step_mask.astype(np.uint16)

        self._covered_field_cells += new_f_cells
        self._covered_obs_cells += new_o_cells
        self._covered_out_cells += new_out_cells
        self._gross_in_field_m2 += gross_in_field_step

        new_area_m2 = new_f_cells * self.cell_area_m2 * self._area_scale
        cov_pct = 100.0 * (self._covered_field_cells / max(1, self.field_cell_count))
        cov_m2_after = self.source_area_m2 * (cov_pct / 100.0)

        # Step-invariant telescoping repeated treatment: sum(delta_R) == max(0, G_T - C_T) identically!
        cum_rep_after = max(0.0, self._gross_in_field_m2 - cov_m2_after)
        rep_step_m2 = cum_rep_after - self._cum_repeated_m2
        self._cum_repeated_m2 = cum_rep_after

        new_out_m2 = new_out_cells * self.cell_area_m2
        new_obs_m2 = new_o_cells * self.cell_area_m2
        untreated_m2 = max(0.0, self.source_area_m2 - cov_m2_after)

        return CoverageStepDelta(
            newly_treated_m2=new_area_m2,
            repeated_treated_m2=rep_step_m2,
            gross_in_field_m2=gross_in_field_step,
            new_outside_field_m2=new_out_m2,
            new_obstacle_intrusion_m2=new_obs_m2,
            coverage_pct_after=cov_pct,
            untreated_m2_after=untreated_m2,
        )

    def snapshot(self) -> CoverageSnapshot:
        cov_pct = 100.0 * (self._covered_field_cells / max(1, self.field_cell_count))
        cov_m2 = self.source_area_m2 * (cov_pct / 100.0)
        untreated_m2 = max(0.0, self.source_area_m2 - cov_m2)
        rep_m2 = max(0.0, self._gross_in_field_m2 - cov_m2)
        rep_pct = (100.0 * rep_m2 / self.source_area_m2) if self.source_area_m2 > 0 else 0.0
        return CoverageSnapshot(
            source_area_m2=self.source_area_m2,
            raster_field_area_m2=self.raster_field_area_m2,
            coverage_pct=cov_pct,
            covered_area_m2=cov_m2,
            untreated_area_m2=untreated_m2,
            gross_treated_in_field_m2=self._gross_in_field_m2,
            repeated_treatment_m2=rep_m2,
            repeated_treatment_pct=rep_pct,
            outside_field_treatment_m2=self._covered_out_cells * self.cell_area_m2,
            obstacle_treatment_m2=self._covered_obs_cells * self.cell_area_m2,
            resolution_m=self.resolution_m,
            memory_bytes=self.memory_bytes,
        )

    def largest_untreated_component_m2(self) -> float:
        """Area (m^2) of the largest 4-connected untreated region inside original field F."""
        untreated = self.in_field & (~self.covered)
        if not np.any(untreated):
            return 0.0
        labeled, n_comp = ndimage_label(untreated)
        if n_comp == 0:
            return 0.0
        counts = np.bincount(labeled.ravel())[1:]
        max_cells = int(np.max(counts)) if len(counts) > 0 else 0
        return max_cells * self.cell_area_m2 * self._area_scale

    def extract_local_map(
        self,
        x: float,
        y: float,
        theta: float,
        *,
        patch_size: int = 24,
        patch_extent_m: float = 36.0,
    ) -> np.ndarray:
        """Extract 5-channel ego-aligned local map (5, H, W) in [0, 1].

        Convention: column axis (+x_loc) = robot forward; row axis (+y_loc) = robot left.
        Channels:
          0: Original agricultural field F
          1: Physical obstacle interiors
          2: Previously treated agricultural area (F & covered)
          3: Remaining untreated agricultural area (F & ~covered)
          4: Permissible robot-center navigable space M
        """
        key = (patch_size, round(patch_extent_m, 4))
        cached = self._local_grid_cache.get(key)
        if cached is None:
            half = 0.5 * patch_extent_m
            lin = np.linspace(-half, half, patch_size, dtype=np.float64)
            lx, ly = np.meshgrid(lin, lin)
            self._local_grid_cache[key] = (lx, ly)
        else:
            lx, ly = cached

        ct, st = math.cos(theta), math.sin(theta)
        wx = x + ct * lx - st * ly
        wy = y + st * lx + ct * ly

        inv_res = 1.0 / self.resolution_m
        ic = np.floor((wx - self.x0) * inv_res).astype(np.int32)
        ir = np.floor((wy - self.y0) * inv_res).astype(np.int32)
        valid = (ic >= 0) & (ic < self.nx) & (ir >= 0) & (ir < self.ny)
        ic_c = np.clip(ic, 0, self.nx - 1)
        ir_c = np.clip(ir, 0, self.ny - 1)

        in_f = valid & self.in_field[ir_c, ic_c]
        in_o = valid & self.in_obstacle[ir_c, ic_c]
        cov = valid & self.covered[ir_c, ic_c]
        nav = valid & self.in_navigable[ir_c, ic_c]

        out = np.empty((5, patch_size, patch_size), dtype=np.float32)
        out[0] = in_f
        out[1] = in_o
        out[2] = in_f & cov
        out[3] = in_f & (~cov)
        out[4] = nav
        return out

    def extract_global_map(
        self,
        x: float,
        y: float,
        theta: float,
        *,
        global_size: int = 32,
        frame: str = "ego_aligned",
    ) -> np.ndarray:
        """Extract aspect-preserving 6-channel global map (6, G, G) in [0, 1].

        Preserves 1:1 physical aspect ratio (no x/y stretching) and uses 2x2 sub-cell
        aggregation so narrow corridors and internal obstacles do not disappear on large fields.

        Frames:
          - "ego_aligned": Centered at field center, rotated by robot heading theta so robot
            forward is ALWAYS +x (same axis convention as local_map!). Span = diag + 4 m so
            field never clips at any rotation angle.
          - "world": World-aligned square bounding box centered at field center with
            span = max(width, height) + 4 m.

        Channels:
          0: Original agricultural field fraction in cell
          1: Physical obstacle presence (max-pooled over 2x2 sub-samples)
          2: Treated agricultural area fraction
          3: Remaining untreated agricultural area fraction
          4: Robot position marker (smooth Gaussian peak at robot cell)
          5: Robot forward-heading marker (smooth peak 1.5 cells ahead along robot heading)
        """
        key = (global_size, frame)
        cached = self._global_grid_cache.get(key)
        if cached is None:
            if frame == "robot_centered":
                span = max(36.0, 2.0 * self.field_diag_m + 4.0)
            elif frame == "ego_aligned":
                span = max(24.0, self.field_diag_m + 4.0)
            else:
                span = max(24.0, max(self.field_width_m, self.field_height_m) + 4.0)
            cell_m = span / global_size
            centers = -0.5 * span + (np.arange(global_size, dtype=np.float64) + 0.5) * cell_m
            use_sub2 = (cell_m > 3.5 * self.resolution_m)
            if use_sub2:
                sub = np.empty(2 * global_size, dtype=np.float64)
                sub[0::2] = centers - 0.25 * cell_m
                sub[1::2] = centers + 0.25 * cell_m
                sx, sy = np.meshgrid(sub, sub)
            else:
                sx, sy = np.meshgrid(centers, centers)
            self._global_grid_cache[key] = (sx, sy, span)
        else:
            sx, sy, span = cached

        use_sub2 = (sx.shape[0] == 2 * global_size)
        cell_m = span / global_size
        if frame == "robot_centered":
            ct, st = math.cos(theta), math.sin(theta)
            wx = x + ct * sx - st * sy
            wy = y + st * sx + ct * sy
            rx_loc = 0.0
            ry_loc = 0.0
            fx_loc = 1.5 * cell_m
            fy_loc = 0.0
        elif frame == "ego_aligned":
            ct, st = math.cos(theta), math.sin(theta)
            wx = self.field_cx + ct * sx - st * sy
            wy = self.field_cy + st * sx + ct * sy
            dx, dy = x - self.field_cx, y - self.field_cy
            rx_loc = ct * dx + st * dy
            ry_loc = -st * dx + ct * dy
            fx_loc = rx_loc + 1.5 * cell_m
            fy_loc = ry_loc
        else:
            wx = self.field_cx + sx
            wy = self.field_cy + sy
            rx_loc = x - self.field_cx
            ry_loc = y - self.field_cy
            fx_loc = rx_loc + 1.5 * cell_m * math.cos(theta)
            fy_loc = ry_loc + 1.5 * cell_m * math.sin(theta)

        inv_res = 1.0 / self.resolution_m
        ic = np.floor((wx - self.x0) * inv_res).astype(np.int32)
        ir = np.floor((wy - self.y0) * inv_res).astype(np.int32)
        valid = (ic >= 0) & (ic < self.nx) & (ir >= 0) & (ir < self.ny)
        ic_c = np.clip(ic, 0, self.nx - 1)
        ir_c = np.clip(ir, 0, self.ny - 1)

        out = np.empty((6, global_size, global_size), dtype=np.float32)
        if use_sub2:
            in_f = (valid & self.in_field[ir_c, ic_c]).reshape(global_size, 2, global_size, 2)
            in_o = (valid & self.in_obstacle_dilated[ir_c, ic_c]).reshape(global_size, 2, global_size, 2)
            cov = (valid & self.covered[ir_c, ic_c]).reshape(global_size, 2, global_size, 2)
            out[0] = in_f.mean(axis=(1, 3))
            out[1] = in_o.max(axis=(1, 3))
            out[2] = (in_f & cov).mean(axis=(1, 3))
            out[3] = (in_f & (~cov)).mean(axis=(1, 3))
        else:
            in_f = valid & self.in_field[ir_c, ic_c]
            in_o = valid & self.in_obstacle_dilated[ir_c, ic_c]
            cov = valid & self.covered[ir_c, ic_c]
            out[0] = in_f
            out[1] = in_o
            out[2] = in_f & cov
            out[3] = in_f & (~cov)

        # Robot position & forward-heading Gaussian markers on (G, G)
        g_lin = -0.5 * span + (np.arange(global_size, dtype=np.float32) + 0.5) * np.float32(cell_m)
        gx_c = g_lin[None, :]
        gy_c = g_lin[:, None]
        inv_2sig2 = np.float32(1.0 / (2.0 * (0.9 * cell_m) ** 2))
        out[4] = np.exp(-((gx_c - np.float32(rx_loc)) ** 2 + (gy_c - np.float32(ry_loc)) ** 2) * inv_2sig2)
        out[5] = np.exp(-((gx_c - np.float32(fx_loc)) ** 2 + (gy_c - np.float32(fy_loc)) ** 2) * inv_2sig2)
        return out

    def extract_global_untreated_cue(self, x: float, y: float, theta: float) -> np.ndarray:
        """Compute 11-D planner-free ego-centric summary of remaining untreated agricultural cells.

        Derived purely from the robot's current treatment state ``in_field & (~covered)``:
          - [0..7]: 8-sector ego-centric angular distribution of remaining untreated area
                    (45 deg wedges centered at 0, +45, +90, +135, 180, -135, -90, -45 deg in
                    robot heading frame; sums to 1.0 whenever untreated area > 0).
          - [8]:    cos(bearing) to nearest untreated cell in robot heading frame (+1 = ahead).
          - [9]:    sin(bearing) to nearest untreated cell in robot heading frame (+1 = left).
          - [10]:   normalized Euclidean distance to nearest untreated cell:
                    clip(dist / max(20.0, field_diag_m), 0.0, 1.0).
        """
        out = np.zeros(11, dtype=np.float32)
        if self._covered_field_cells >= self.field_cell_count:
            return out

        # Use coarse 2x strided grid when plenty of untreated cells remain (~35 us);
        # switch to full-resolution field cells once < 15% remains so tiny pockets are never missed!
        rem_frac = 1.0 - (self._covered_field_cells / max(1, self.field_cell_count))
        if rem_frac > 0.15:
            untr_m = ~self.covered[self._coarse_ir, self._coarse_ic]
            ux = self._coarse_wx[untr_m]
            uy = self._coarse_wy[untr_m]
        else:
            untr_m = ~self.covered[self._field_ir, self._field_ic]
            ux = self._field_wx[untr_m]
            uy = self._field_wy[untr_m]
        if ux.size == 0:
            untr_m = ~self.covered[self._field_ir, self._field_ic]
            ux = self._field_wx[untr_m]
            uy = self._field_wy[untr_m]
            if ux.size == 0:
                return out

        dx = ux - np.float32(x)
        dy = uy - np.float32(y)
        ct = np.float32(math.cos(theta))
        st = np.float32(math.sin(theta))
        # Rotate into robot ego frame (+lx = forward, +ly = left)
        lx = ct * dx + st * dy
        ly = -st * dx + ct * dy
        d2 = lx * lx + ly * ly

        # 8-sector angular histogram centered at 0, 45, ..., 315 deg
        ang = np.arctan2(ly, lx)  # [-pi, pi]
        shifted = np.mod(ang + np.float32(math.pi / 8.0), np.float32(2.0 * math.pi))
        bins = np.floor(shifted * np.float32(4.0 / math.pi)).astype(np.int32) & 7
        counts = np.bincount(bins, minlength=8).astype(np.float32)
        out[:8] = counts / np.float32(ux.size)

        # Nearest untreated cell bearing & normalized distance
        idx_min = int(np.argmin(d2))
        dist_min = float(math.sqrt(float(d2[idx_min])))
        if dist_min > 1e-6:
            out[8] = float(lx[idx_min]) / dist_min
            out[9] = float(ly[idx_min]) / dist_min
        else:
            out[8] = 1.0
            out[9] = 0.0
        out[10] = min(1.0, dist_min / max(20.0, self.field_diag_m))
        return out


def compute_exact_reference_coverage(
    field_polygon: Polygon,
    trajectory_xy: np.ndarray | list[tuple[float, float]],
    working_width_m: float,
    *,
    chunk_length_m: float = 20.0,
) -> dict[str, float]:
    """Exact high-accuracy Shapely geometric coverage evaluator for scientific verification."""
    pts = np.asarray(trajectory_xy, dtype=np.float64)[:, :2]
    src_area = float(field_polygon.area)
    empty = {
        "source_area_m2": src_area,
        "coverage_pct": 0.0,
        "covered_area_m2": 0.0,
        "untreated_area_m2": src_area,
        "repeated_treatment_m2": 0.0,
        "repeated_treatment_pct": 0.0,
        "outside_field_treatment_m2": 0.0,
        "obstacle_treatment_m2": 0.0,
        "largest_untreated_component_m2": src_area,
    }
    if len(pts) < 2:
        return empty

    d = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    keep = np.concatenate([[True], d > 1e-6])
    pts = pts[keep]
    if len(pts) < 2:
        return empty

    half_w = 0.5 * float(working_width_m)
    ext_poly = Polygon(field_polygon.exterior)
    hole_polys = [Polygon(r) for r in field_polygon.interiors]
    holes_u = shapely.union_all(hole_polys) if hole_polys else Polygon()

    strips: list[BaseGeometry] = []
    sum_in_f = 0.0
    cur_start = 0
    acc_len = 0.0
    for i in range(1, len(pts)):
        acc_len += math.hypot(float(pts[i, 0] - pts[i - 1, 0]), float(pts[i, 1] - pts[i - 1, 1]))
        if acc_len >= chunk_length_m or i == len(pts) - 1:
            sub = pts[cur_start : i + 1]
            if len(sub) >= 2:
                st = LineString(sub).buffer(half_w, cap_style="flat", join_style="round", quad_segs=12)
                if not st.is_empty and st.area > 0.0:
                    strips.append(st)
                    sum_in_f += float(_areal(st.intersection(field_polygon)).area)
            cur_start = i
            acc_len = 0.0

    if not strips:
        return empty

    union_fp = _areal(shapely.union_all(strips, grid_size=1e-4))
    cov_in_f_geom = _areal(union_fp.intersection(field_polygon))
    cov_in_f = float(cov_in_f_geom.area)
    cov_pct = 100.0 * (cov_in_f / src_area) if src_area > 0 else 0.0
    rep_m2 = max(0.0, sum_in_f - cov_in_f)
    out_m2 = float(_areal(union_fp.difference(ext_poly)).area)
    obs_m2 = float(_areal(union_fp.intersection(holes_u)).area) if not holes_u.is_empty else 0.0

    rem_geom = _areal(field_polygon.difference(cov_in_f_geom))
    if rem_geom.is_empty:
        largest_rem = 0.0
    else:
        parts = [p.area for p in shapely.get_parts(rem_geom) if p.geom_type == "Polygon"]
        largest_rem = float(max(parts)) if parts else 0.0

    return {
        "source_area_m2": src_area,
        "coverage_pct": cov_pct,
        "covered_area_m2": cov_in_f,
        "untreated_area_m2": max(0.0, src_area - cov_in_f),
        "repeated_treatment_m2": rep_m2,
        "repeated_treatment_pct": (100.0 * rep_m2 / src_area) if src_area > 0 else 0.0,
        "outside_field_treatment_m2": out_m2,
        "obstacle_treatment_m2": obs_m2,
        "largest_untreated_component_m2": largest_rem,
    }
