"""Verify the packaged corpus, preserved smoke fields and curriculum bindings."""
from collections import Counter
import json
from pathlib import Path
import math
import numpy as np
from shapely import normalize
from shapely.geometry import Polygon
from session import check_data, source_signature
from ppo_coverage_planning_v1 import AgriculturalFieldDataset
from ppo_coverage_planning_v1.curriculum import sample_task_for_stage
from ppo_coverage_planning_v1.datasets import _canonical_geom_hash


def main():
    directory = Path(__file__).resolve().parent / "data"
    path = directory / "fields.json"
    original = directory / "smoke8" / "fields.json"
    print("Source", source_signature(), "dataset", check_data(path))
    check_data(original)
    old = json.loads(original.read_bytes())["fields"]
    rows = json.loads(path.read_bytes())["fields"]
    by_id = {r["field_id"]: r for r in rows}
    assert len(old) == 8 and all(by_id[r["field_id"]] == r for r in old)
    dataset = AgriculturalFieldDataset(path)
    records = dataset.records
    expected = {(f, s): n for f in "ABCD" for s, n in (("train", 2560), ("val", 320), ("test", 320))}
    assert Counter((r.family, r.split) for r in records) == expected
    assert len({r.geom_sha256 for r in records}) == len(by_id) == len(records)
    signatures = set()
    for r in records:
        p = r.to_polygon()
        assert r.source == "synthetic" and p.is_valid and p.area > 0
        assert len(p.interiors) == r.obstacle_count
        assert math.isclose(p.area, r.area_m2, abs_tol=0.00051)
        assert _canonical_geom_hash(p) == r.geom_sha256
        convex = Polygon(p.exterior).equals(Polygon(p.exterior).convex_hull)
        assert convex == (r.family in "AC") and bool(p.interiors) == (r.family in "CD")
        signatures.add(normalize(p).wkb)
    assert len(signatures) == len(records)
    for split in ("train", "val"):
        rng = np.random.default_rng(42)
        observed = set()
        for _ in range(256):
            field, _ = sample_task_for_stage(6, rng, dataset=dataset, split=split)
            if field.field_id in by_id:
                assert by_id[field.field_id]["split"] == split
                observed.add(field.family)
            if observed == set("ABCD"):
                break
        assert observed == set("ABCD")
    print("Verified all 12,800 fields, three splits, eight anchors and stage-6 curriculum access.")


if __name__ == "__main__":
    main()
