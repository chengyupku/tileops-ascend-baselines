"""Import the reviewed development snapshot; never run a device benchmark."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from ascend950_data import LEGACY_PROTOCOL, key, require, validate_performance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/ascend950/development-v2.json"))
    args = parser.parse_args()
    root = args.workspace
    reviewed = root / "docs/ascend950/dev/resume_20261020_profile_guided"
    baseline = root / "docs/ascend950/results/remeasure_20261018_v2"
    sources = {}

    def read(path):
        raw = path.read_bytes()
        sources[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        return raw.decode()

    rows = [json.loads(line) for line in read(reviewed / "tileops_latest.jsonl").splitlines()]
    comparisons = {key(r): r for r in csv.DictReader(read(reviewed / "comparison_latest.csv").splitlines())}
    best = {key(r): r for r in csv.DictReader(read(baseline / "best.csv").splitlines())}
    originals = {}
    for name in ("reference.jsonl", "candidates.jsonl"):
        for line in read(baseline / name).splitlines():
            row = json.loads(line)
            if row.get("status") in ("measured_reference", "measured_candidate"):
                originals.setdefault((*key(row), row["candidate"]), []).append(row)

    def measurement(row):
        require(row["measurement_protocol"] == LEGACY_PROTOCOL, "Mixed protocols")
        require("950" in row["environment"]["device_name"], "Mixed hardware")
        require(row["profiler"] == "torch_npu.profiler.profile", "Wrong timer")
        require(not row.get("input_mutated") and not row.get("noncomparable_reason"), "Invalid comparison")
        if row["candidate"] != "manifest_reference":
            require(row["checked_outputs"] == row["total_outputs"] > 0, "Unchecked outputs")
        return {
            "candidate": row["candidate"], "latency_ms": row["latency_ms"],
            "samples_ms": row["latency_samples_ms"], "warmup": row["warmup"],
            "repeats": row["repeats"], "measured_at": row["recorded_at_utc"],
            "run_id": row["run_id"], "toolchain_sha256": row["toolchain_sha256"],
            "tilelang_commit": row["environment"]["tilelang_head"],
            "tileops_commit_at_measurement": row["environment"]["tileops_head"],
        }

    cases = []
    for row in rows:
        k = key(row)
        winner = best[k]
        matches = [r for r in originals[(*k, winner["best_baseline"]) ]
                   if r["params"] == row["params"]
                   and math.isclose(r["latency_ms"], float(winner["best_latency_ms"]), rel_tol=1e-12)]
        require(bool(matches), f"No raw evidence for selected baseline: {k}")
        ours, rival = measurement(row), measurement(matches[-1])
        ratio = rival["latency_ms"] / ours["latency_ms"]
        require(math.isclose(float(comparisons[k]["baseline_over_kernel"]), ratio, rel_tol=1e-12),
                f"Reviewed comparison mismatch: {k}")
        cases.append({"op": row["op"], "case_id": row["case_id"], "family": row["family"],
                      "params": row["params"], "tileops": ours, "baseline": rival,
                      "ratio": ratio, "passes_80pct": ratio >= 0.8})
    require({key(r) for r in rows} == set(comparisons), "Comparison coverage differs")
    result = validate_performance({
        "schema": "ascend950-development/v1", "hardware": "Ascend950DT_9592",
        "protocol": LEGACY_PROTOCOL, "clock_locked": False, "release_score_eligible": False,
        "metric": "mean_sum_device_kernel_ms", "flush_bytes": 1 << 30,
        "baseline_scope": "Fastest eligible candidate in the frozen 2026-10-18 v2 pool; additional library integration remains ongoing.",
        "source_sha256": sources, "cases": sorted(cases, key=key),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"Imported {len(cases)} measured cases across {len({r['op'] for r in cases})} operators")


if __name__ == "__main__":
    main()
