"""Extract the project's registered real-field cohort from the publisher's GeoParquet."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import pyarrow.parquet as pq
from pyproj import Transformer
from shapely import wkb
from shapely.ops import transform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ppo_coverage_planning_v1.datasets import _make_centered_record


def prepare(source, dest):
    selection = json.loads(Path(__file__).with_name("eu94_selection.json").read_bytes())
    targets = sorted(selection, key=lambda r: r["source_row"])
    parquet = pq.ParquetFile(source)
    if parquet.metadata.num_rows != 51398853:
        raise ValueError("Expected the original Zenodo field_boundaries.parquet")
    results = {}
    offset = pos = 0
    for group in range(parquet.num_row_groups):
        size = parquet.metadata.row_group(group).num_rows
        if pos == len(targets):
            break
        if targets[pos]["source_row"] >= offset + size:
            offset += size
            continue
        for batch in parquet.iter_batches(batch_size=8192, row_groups=[group], columns=["id", "geometry"]):
            while pos < len(targets) and targets[pos]["source_row"] < offset + len(batch):
                row = targets[pos]
                idx = row["source_row"] - offset
                raw = batch.column(1)[idx].as_py()
                fid = row["field_id"]
                if str(batch.column(0)[idx].as_py()) != fid or hashlib.sha256(raw).hexdigest() != row["source_sha256"]:
                    raise ValueError(f"Source identity mismatch for {fid}. Check the archive checksum.")
                project = Transformer.from_crs("EPSG:4326", row["metric_crs"], always_xy=True)
                poly = transform(project.transform, wkb.loads(raw))
                record = _make_centered_record(poly, field_id=fid, family=row["family"], source="real_eu",
                        split=row["split"], metadata={"metric_crs":row["metric_crs"], "source_row":row["source_row"]})
                if (not poly.is_valid or record.geom_sha256 != row["geom_sha256"] or
                        len(poly.interiors) != row["holes"] or not math.isclose(poly.area, row["area_m2"], abs_tol=0.00051)):
                    raise ValueError(f"Metric geometry mismatch for {fid}")
                results[fid] = record.to_dict()
                pos += 1
            offset += len(batch)
    records = [results[r["field_id"]] for r in selection]
    dest.mkdir(parents=True, exist_ok=False)
    output = dest / "real_fields_splits.json"
    output.write_text(json.dumps({"fields":records, "source":"https://zenodo.org/records/14229033",
        "attribution":"Planet and Sinergise, EU field boundaries", "license":"CC BY-SA 4.0",
        "changes":"Registered subset, A–D labels, split assignments, projection to local metric coordinates and centering."},
        sort_keys=True) + "\n")
    (dest / "checksums.json").write_text(json.dumps({output.name:hashlib.sha256(output.read_bytes()).hexdigest()})+"\n")
    print(output, dict(Counter(r["split"] for r in records)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare(args.parquet, args.destination)
