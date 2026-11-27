"""Verify the bundled synthetic geometry and curriculum bindings."""
from collections import Counter
from pathlib import Path
import math
import numpy as np
from session import check_data, source_signature
from ppo_coverage_planning_v1 import AgriculturalFieldDataset
from ppo_coverage_planning_v1.curriculum import sample_task_for_stage
from ppo_coverage_planning_v1.datasets import _canonical_geom_hash


def main():
    path = Path(__file__).parent / "data" / "fields.json"
    print("Source", source_signature(), "dataset", check_data(path))
    dataset = AgriculturalFieldDataset(path)
    records = dataset.records
    expected = Counter((family, split) for family in "ABCD" for split in ("train", "val"))
    assert Counter((r.family, r.split) for r in records) == expected
    assert len({r.geom_sha256 for r in records}) == len({r.field_id for r in records}) == 8
    for record in records:
        poly = record.to_polygon()
        assert record.source == "synthetic" and poly.is_valid and poly.area > 0
        assert len(poly.interiors) == record.obstacle_count
        assert math.isclose(poly.area, record.area_m2, abs_tol=0.00051)
        assert _canonical_geom_hash(poly) == record.geom_sha256
    for split in ("train", "val"):
        rng = np.random.default_rng(42)
        allowed = {r.field_id for r in records if r.split == split}
        observed = set()
        for _ in range(128):
            field, _ = sample_task_for_stage(6, rng, dataset=dataset, split=split)
            if field.field_id.startswith("procedural-"):
                observed.add(field.field_id)
            if observed == allowed:
                break
        assert observed == allowed
    print("Verified 8 polygons, separate training and validation splits, and curriculum access.")


if __name__ == "__main__":
    main()
