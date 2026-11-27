"""Rebuild the 12,800-field synthetic training corpus with its original eight anchors."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from shapely.geometry import Polygon
from ppo_coverage_planning_v1 import datasets as generators

ROOT = Path(__file__).resolve().parent / "data"
SEED = 20261127
COUNTS = {"train": 2560, "val": 320, "test": 320}


def family_of(poly):
    shell = Polygon(poly.exterior)
    convex = shell.equals(shell.convex_hull)
    return ("C" if convex else "D") if poly.interiors else ("A" if convex else "B")


def draw(family, rng, field_id):
    if family == "A":
        return generators.generate_synthetic_convex(rng, field_id=field_id)
    if family == "B":
        return generators.generate_synthetic_nonconvex(rng, field_id=field_id)
    return generators.generate_synthetic_with_obstacles(
        rng, nonconvex_exterior=family == "D", n_obstacles=int(rng.integers(1, 4)), field_id=field_id)


def build():
    start = time.monotonic()
    rows = json.loads((ROOT / "smoke8/fields.json").read_bytes())["fields"]
    seen = {r["geom_sha256"] for r in rows}
    for i, family in enumerate("ABCD"):
        for j, (split, count) in enumerate(COUNTS.items()):
            rng = np.random.default_rng([SEED, i, j])
            existing = sum(r["family"] == family and r["split"] == split for r in rows)
            for k in range(existing, count):
                field_id = f"synthetic-{family}-{split}-{k:04d}"
                while True:
                    rec = draw(family, rng, field_id)
                    poly = rec.to_polygon()
                    if poly.is_valid and poly.area > 0 and family_of(poly) == family and rec.geom_sha256 not in seen:
                        break
                seen.add(rec.geom_sha256)
                row = rec.to_dict()
                row["split"] = split
                rows.append(row)
    path = ROOT / "fields.json"
    doc = {"version": "synthetic-12800-v1", "units": "local Cartesian metres",
           "seed": SEED, "counts_per_family": COUNTS, "fields": rows}
    path.write_text(json.dumps(doc, separators=(",", ":"), sort_keys=True) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT / "checksums.json").write_text(json.dumps({path.name: digest}, indent=2) + "\n")
    print(f"Generated {len(rows)} fields in {time.monotonic()-start:.2f}s", flush=True)


if __name__ == "__main__":
    build()
