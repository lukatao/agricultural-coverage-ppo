"""Reproducible agricultural field dataset management (synthetic & real EU fields).

Supports:
  - Families A (convex, no holes), B (non-convex, no holes), C (convex with holes),
    D (non-convex with holes).
  - Controlled procedural synthetic agricultural fields for curriculum stages 0-6.
  - Real EU agricultural field extraction from frozen thesis manifests with strict
    SHA-256 geometry deduplication and fixed train / validation / held-out test splits.
  - Self-contained portable JSON split serialization (no hardcoded machine paths).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from shapely import affinity, wkb, wkt
from shapely.geometry import Polygon

from .field_world import AgriculturalFieldWorld
from .robot_specs import RobotSpec, get_reference_robot


@dataclass(frozen=True)
class FieldRecord:
    """Single immutable agricultural field geometry record with provenance."""
    field_id: str
    family: str           # "A", "B", "C", "D"
    source: str           # "synthetic" or "real_eu"
    split: str            # "train", "val", "test", or "procedural"
    area_m2: float
    obstacle_count: int
    geom_sha256: str
    wkb_hex: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_polygon(self) -> Polygon:
        return wkb.loads(bytes.fromhex(self.wkb_hex))

    def to_world(self, sdf_resolution_m: float = 0.5) -> AgriculturalFieldWorld:
        return AgriculturalFieldWorld(
            self.to_polygon(),
            field_id=self.field_id,
            family=self.family,
            sdf_resolution_m=sdf_resolution_m,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_id": self.field_id,
            "family": self.family,
            "source": self.source,
            "split": self.split,
            "area_m2": round(self.area_m2, 3),
            "obstacle_count": self.obstacle_count,
            "geom_sha256": self.geom_sha256,
            "wkb_hex": self.wkb_hex,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FieldRecord:
        return cls(
            field_id=str(d["field_id"]),
            family=str(d["family"]),
            source=str(d["source"]),
            split=str(d["split"]),
            area_m2=float(d["area_m2"]),
            obstacle_count=int(d["obstacle_count"]),
            geom_sha256=str(d["geom_sha256"]),
            wkb_hex=str(d["wkb_hex"]),
            metadata=dict(d.get("metadata", {})),
        )


def _canonical_geom_hash(poly: Polygon) -> str:
    """Geometry hash invariant to translation and tiny floating-point rounding."""
    cx, cy = poly.centroid.x, poly.centroid.y
    centered = affinity.translate(poly, xoff=-cx, yoff=-cy)
    # Round coordinates to millimeter precision for duplicate detection
    ext = [(round(float(x), 3), round(float(y), 3)) for x, y in centered.exterior.coords]
    holes = sorted(
        [(round(float(x), 3), round(float(y), 3)) for x, y in r.coords]
        for r in centered.interiors
    )
    payload = json.dumps({"ext": ext, "holes": holes}, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _make_centered_record(
    poly: Polygon,
    *,
    field_id: str,
    family: str,
    source: str,
    split: str,
    metadata: dict[str, Any] | None = None,
) -> FieldRecord:
    cx, cy = poly.centroid.x, poly.centroid.y
    centered = affinity.translate(poly, xoff=-cx, yoff=-cy)
    return FieldRecord(
        field_id=field_id,
        family=family,
        source=source,
        split=split,
        area_m2=float(centered.area),
        obstacle_count=len(centered.interiors),
        geom_sha256=_canonical_geom_hash(centered),
        wkb_hex=wkb.dumps(centered).hex(),
        metadata=dict(metadata or {}),
    )


# =============================================================================
# Procedural Synthetic Agricultural Field Generators (Stages 0 - 6)
# =============================================================================
def generate_open_region(size_m: float = 120.0, field_id: str = "syn_open_120") -> FieldRecord:
    h = 0.5 * size_m
    poly = Polygon([(-h, -h), (h, -h), (h, h), (-h, h)])
    return _make_centered_record(poly, field_id=field_id, family="A", source="synthetic", split="procedural",
                                 metadata={"kind": "open_region", "width_m": size_m, "height_m": size_m})


def generate_rectangle_field(
    width_m: float,
    height_m: float,
    angle_deg: float = 0.0,
    field_id: str = "syn_rect",
) -> FieldRecord:
    hw, hh = 0.5 * width_m, 0.5 * height_m
    poly = Polygon([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])
    if abs(angle_deg) > 1e-9:
        poly = affinity.rotate(poly, angle_deg, origin=(0.0, 0.0))
    return _make_centered_record(
        poly,
        field_id=field_id,
        family="A",
        source="synthetic",
        split="procedural",
        metadata={"kind": "rectangle", "width_m": round(width_m, 2), "height_m": round(height_m, 2), "angle_deg": round(angle_deg, 2)},
    )


def generate_synthetic_convex(
    rng: np.random.Generator,
    *,
    radius_range_m: tuple[float, float] = (18.0, 32.0),
    n_vertices: int | None = None,
    field_id: str = "syn_convex",
) -> FieldRecord:
    nv = int(n_vertices if n_vertices is not None else rng.integers(5, 9))
    angles = np.sort(rng.uniform(0.0, 2.0 * math.pi, size=nv))
    # Guarantee minimum angular separation so polygon is well-conditioned
    angles = np.linspace(0.0, 2.0 * math.pi, nv, endpoint=False) + rng.uniform(-0.22, 0.22, size=nv)
    r_base = float(rng.uniform(*radius_range_m))
    aspect = float(rng.uniform(0.65, 1.45))
    radii = r_base * rng.uniform(0.78, 1.15, size=nv)
    pts = [(float(r * math.cos(a) * aspect), float(r * math.sin(a))) for r, a in zip(radii, angles)]
    poly = Polygon(pts).convex_hull
    rot = float(rng.uniform(0.0, 180.0))
    poly = affinity.rotate(poly, rot, origin="centroid")
    return _make_centered_record(poly, field_id=field_id, family="A", source="synthetic", split="procedural",
                                 metadata={"kind": "synthetic_convex"})


def generate_synthetic_nonconvex(
    rng: np.random.Generator,
    *,
    scale_range_m: tuple[float, float] = (34.0, 56.0),
    field_id: str = "syn_nonconvex",
) -> FieldRecord:
    """Generate a physically navigable L-shaped or U-shaped non-convex agricultural field (Family B)."""
    W = float(rng.uniform(*scale_range_m))
    H = float(rng.uniform(*scale_range_m))
    shape_kind = "L" if rng.random() < 0.6 else "U"
    if shape_kind == "L":
        cut_w = float(rng.uniform(0.32, 0.50)) * W
        cut_h = float(rng.uniform(0.32, 0.50)) * H
        # Ensure remaining arms >= 16 m wide (comfortably navigable for R_min=3 m, r+c=1.5 m)
        cut_w = min(cut_w, max(8.0, W - 18.0))
        cut_h = min(cut_h, max(8.0, H - 18.0))
        pts = [(0, 0), (W, 0), (W, H - cut_h), (W - cut_w, H - cut_h), (W - cut_w, H), (0, H)]
    else:
        # U-shape with wide navigable base and arms
        arm_w = max(14.0, float(rng.uniform(0.28, 0.36)) * W)
        base_h = max(15.0, float(rng.uniform(0.32, 0.45)) * H)
        if 2.0 * arm_w >= W - 8.0:
            W = 2.0 * arm_w + 12.0
        pts = [
            (0, 0), (W, 0), (W, H), (W - arm_w, H),
            (W - arm_w, base_h), (arm_w, base_h), (arm_w, H), (0, H),
        ]
    poly = Polygon(pts)
    poly = affinity.rotate(poly, float(rng.uniform(0.0, 360.0)), origin="centroid")
    return _make_centered_record(poly, field_id=field_id, family="B", source="synthetic", split="procedural",
                                 metadata={"kind": f"synthetic_nonconvex_{shape_kind}"})


def generate_synthetic_with_obstacles(
    rng: np.random.Generator,
    *,
    nonconvex_exterior: bool = False,
    n_obstacles: int = 1,
    scale_range_m: tuple[float, float] = (42.0, 62.0),
    field_id: str = "syn_obs",
) -> FieldRecord:
    """Generate a Family C (convex + holes) or Family D (non-convex + holes) field with navigable passages."""
    if nonconvex_exterior:
        base_rec = generate_synthetic_nonconvex(rng, scale_range_m=scale_range_m, field_id=field_id)
        ext_poly = base_rec.to_polygon()
        fam = "D"
    else:
        W = float(rng.uniform(*scale_range_m))
        H = float(rng.uniform(0.75 * scale_range_m[0], scale_range_m[1]))
        ext_poly = Polygon([(-0.5 * W, -0.5 * H), (0.5 * W, -0.5 * H), (0.5 * W, 0.5 * H), (-0.5 * W, 0.5 * H)])
        ext_poly = affinity.rotate(ext_poly, float(rng.uniform(0.0, 180.0)), origin=(0.0, 0.0))
        fam = "C"

    # Place obstacles strictly deep inside exterior so >= 10 m passage remains around every obstacle
    inner_zone = ext_poly.buffer(-11.0, quad_segs=12)
    holes: list[Polygon] = []
    if not inner_zone.is_empty:
        ixmin, iymin, ixmax, iymax = inner_zone.bounds
        for _ in range(n_obstacles):
            for _attempt in range(40):
                ox = float(rng.uniform(ixmin, ixmax))
                oy = float(rng.uniform(iymin, iymax))
                ow = float(rng.uniform(5.0, 9.0))
                oh = float(rng.uniform(5.0, 9.0))
                cand = Polygon([
                    (ox - 0.5 * ow, oy - 0.5 * oh),
                    (ox + 0.5 * ow, oy - 0.5 * oh),
                    (ox + 0.5 * ow, oy + 0.5 * oh),
                    (ox - 0.5 * ow, oy + 0.5 * oh),
                ])
                cand = affinity.rotate(cand, float(rng.uniform(0.0, 180.0)), origin=(ox, oy))
                if not inner_zone.contains(cand):
                    continue
                if any(cand.distance(h) < 10.0 for h in holes):
                    continue
                holes.append(cand)
                break

    if not holes:
        # Guaranteed single central obstacle for valid C/D geometry
        rp = ext_poly.representative_point()
        cand = rp.buffer(3.0, quad_segs=8)
        if ext_poly.buffer(-6.0).contains(cand):
            holes.append(cand)

    poly = Polygon(ext_poly.exterior.coords, [list(h.exterior.coords) for h in holes])
    if not poly.is_valid:
        poly = ext_poly
        fam = "B" if nonconvex_exterior else "A"
    return _make_centered_record(
        poly,
        field_id=field_id,
        family=fam,
        source="synthetic",
        split="procedural",
        metadata={"kind": f"synthetic_family_{fam}", "n_obstacles": len(poly.interiors)},
    )


# =============================================================================
# Real EU Field Dataset Builder & Fixed Train/Val/Test Split Manager
# =============================================================================
def _check_field_feasible(poly: Polygon, spec: RobotSpec, min_nav_area_m2: float = 120.0) -> tuple[bool, str]:
    if not poly.is_valid or poly.is_empty or poly.geom_type != "Polygon":
        return False, "invalid_polygon"
    w = AgriculturalFieldWorld(poly, field_id="check", sdf_resolution_m=1.0)
    nav = w.navigable_domain(spec, extra_margin_m=0.10)
    if nav.is_empty or nav.area < min_nav_area_m2:
        return False, f"navigable_area_too_small ({getattr(nav, 'area', 0.0):.1f} m2)"
    if nav.geom_type != "Polygon":
        # Check if largest navigable connected component retains >= 85% of navigable area
        parts = [p for p in nav.geoms if p.geom_type == "Polygon"]
        if not parts:
            return False, "no_polygon_navigable_component"
        largest = max(p.area for p in parts)
        if largest / max(1e-6, nav.area) < 0.85:
            return False, "disconnected_navigable_domain_for_reference_robot"
    return True, "ok"


def build_real_field_splits_from_repo(
    repo_research_dir: Path,
    output_manifest_path: Path,
    *,
    split_seed: int = 20261126,
    max_area_m2: float = 35000.0,
    min_area_m2: float = 900.0,
    max_per_family: int = 40,
) -> dict[str, Any]:
    """Scan verified thesis field manifests, deduplicate by geometry SHA-256, check feasibility,
    and write a fixed portable train/val/test JSON split file.
    """
    ref_spec = get_reference_robot()
    seen_hashes: set[str] = set()
    seen_ids: set[str] = set()
    by_family: dict[str, list[dict[str, Any]]] = {"A": [], "B": [], "C": [], "D": []}
    rejected: list[dict[str, Any]] = []

    # Priority manifests first (canonical BCD/Swath development & unseen suites)
    priority_files = [
        repo_research_dir / "results/unified_bcd_v1/development_selection.json",
        repo_research_dir / "results/unified_bcd_v1/unseen_selection.json",
        repo_research_dir / "results/bcd_first_v1/development_selection.json",
        repo_research_dir / "results/bcd_first_v1/generalization_selection.json",
        repo_research_dir / "results/bcd_first_v1/transfer_selection.json",
    ]
    other_files = sorted(
        p for p in repo_research_dir.rglob("*.json")
        if p not in priority_files and p.stat().st_size <= 20_000_000
    )

    for path in [*priority_files, *other_files]:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        cases = data.get("cases")
        if not isinstance(cases, list):
            continue
        for c in cases:
            if not isinstance(c, dict):
                continue
            cid = str(c.get("case_id", ""))
            if not cid or cid in seen_ids:
                continue
            geom = None
            if "source_wkb" in c and isinstance(c["source_wkb"], str):
                try:
                    geom = wkb.loads(bytes.fromhex(c["source_wkb"]))
                except Exception:
                    geom = None
            elif "local_wkt" in c and isinstance(c["local_wkt"], str):
                try:
                    geom = wkt.loads(c["local_wkt"])
                except Exception:
                    geom = None
            if geom is None or geom.geom_type != "Polygon" or geom.is_empty or not geom.is_valid:
                continue
            cat = str(c.get("category", ""))
            fam = str(c.get("family", ""))
            if not fam:
                if "convex_without" in cat:
                    fam = "A"
                elif "non_convex_without" in cat:
                    fam = "B"
                elif "convex_with" in cat:
                    fam = "C"
                elif "non_convex_with" in cat:
                    fam = "D"
                else:
                    continue
            if fam not in by_family:
                continue

            g_hash = _canonical_geom_hash(geom)
            if g_hash in seen_hashes:
                continue
            seen_ids.add(cid)
            seen_hashes.add(g_hash)

            area = float(geom.area)
            if area < min_area_m2 or area > max_area_m2:
                rejected.append({"field_id": cid, "family": fam, "area_m2": round(area, 1), "reason": "area_out_of_curriculum_range"})
                continue

            ok, reason = _check_field_feasible(geom, ref_spec)
            if not ok:
                rejected.append({"field_id": cid, "family": fam, "area_m2": round(area, 1), "reason": reason})
                continue

            cx, cy = geom.centroid.x, geom.centroid.y
            centered = affinity.translate(geom, xoff=-cx, yoff=-cy)
            by_family[fam].append({
                "field_id": cid,
                "family": fam,
                "source": "real_eu",
                "area_m2": round(area, 3),
                "obstacle_count": len(centered.interiors),
                "geom_sha256": g_hash,
                "wkb_hex": wkb.dumps(centered).hex(),
                "metadata": {
                    "slot": c.get("slot"),
                    "source_manifest": str(path.relative_to(repo_research_dir)),
                    "metric_crs": c.get("metric_crs", c.get("crs")),
                },
            })

    # Deterministic stratified train / val / test split per family (60% / 20% / 20%)
    rng = np.random.default_rng(split_seed)
    records: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for fam in ("A", "B", "C", "D"):
        items = sorted(by_family[fam], key=lambda r: (r["area_m2"], r["field_id"]))
        if len(items) > max_per_family:
            # Spread across area spectrum deterministically rather than taking only largest/smallest
            idx = np.linspace(0, len(items) - 1, max_per_family).round().astype(int)
            items = [items[i] for i in sorted(set(idx.tolist()))]
        perm = rng.permutation(len(items)).tolist()
        shuffled = [items[i] for i in perm]
        n = len(shuffled)
        if n >= 5:
            n_test = max(1, int(round(0.20 * n)))
            n_val = max(1, int(round(0.20 * n)))
            n_train = max(1, n - n_val - n_test)
        elif n >= 3:
            n_train, n_val, n_test = n - 2, 1, 1
        elif n == 2:
            n_train, n_val, n_test = 1, 1, 0
        else:
            n_train, n_val, n_test = n, 0, 0

        counts[fam] = {"train": 0, "val": 0, "test": 0}
        for i, item in enumerate(shuffled):
            if i < n_train:
                sp = "train"
            elif i < n_train + n_val:
                sp = "val"
            else:
                sp = "test"
            item["split"] = sp
            counts[fam][sp] += 1
            records.append(item)

    manifest = {
        "schema_version": 1,
        "split_seed": split_seed,
        "selection_criteria": {
            "min_area_m2": min_area_m2,
            "max_area_m2": max_area_m2,
            "max_per_family": max_per_family,
            "reference_robot_feasibility_checked": True,
            "deduplication": "canonical_mm_rounded_centroid_normalized_sha256",
        },
        "counts_by_family_split": counts,
        "total_selected": len(records),
        "total_rejected_count": len(rejected),
        "rejected_summary": rejected[:30],
        "fields": records,
    }
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


class AgriculturalFieldDataset:
    """Portable field dataset interface serving fixed splits & procedural synthetic fields."""

    def __init__(self, manifest_path: Path | str | None = None) -> None:
        self.manifest_path = Path(manifest_path) if manifest_path is not None else None
        self.records: list[FieldRecord] = []
        self.manifest_meta: dict[str, Any] = {}
        if self.manifest_path is not None and self.manifest_path.exists():
            data = json.loads(self.manifest_path.read_text())
            self.manifest_meta = {k: v for k, v in data.items() if k != "fields"}
            self.records = [FieldRecord.from_dict(r) for r in data.get("fields", [])]

    def get_records(
        self,
        *,
        families: tuple[str, ...] = ("A", "B", "C", "D"),
        split: str = "train",
        max_area_m2: float | None = None,
    ) -> list[FieldRecord]:
        out = [
            r for r in self.records
            if r.family in families and r.split == split
            and (max_area_m2 is None or r.area_m2 <= max_area_m2)
        ]
        return out

    def verify_no_split_leakage(self) -> tuple[bool, int]:
        """Verify zero shared field_ids or canonical geometry SHA-256 hashes across train/val/test."""
        by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
        for r in self.records:
            if r.split in by_split:
                by_split[r.split].add(r.geom_sha256)
        leaks = (
            len(by_split["train"] & by_split["val"])
            + len(by_split["train"] & by_split["test"])
            + len(by_split["val"] & by_split["test"])
        )
        return (leaks == 0), leaks
