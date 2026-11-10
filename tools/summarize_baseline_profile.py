#!/usr/bin/env python3
"""Summarize a resumable baseline profile JSONL without treating failures as latency."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.baseline_protocol import PROTOCOL_ID, TERMINAL_FAILURES, eligible


GOOD = {
    "measured",
    "measured_reference",
    "measured_reference_input_mutated",
    "measured_candidate",
    "measured_candidate_input_mutated",
}
# Finding 3: a candidate that failed workloads.numerics.verify() is a terminal outcome (it
# will not be silently retried), but it is deliberately excluded from GOOD so it can never
# win best_baseline on latency alone.
TERMINAL = GOOD | TERMINAL_FAILURES | {"invalidated_external_activity", "candidate_inventory"}

# Finding 4: the documented measurement contract (docs/ascend950/baseline_measurement.md).
# A row sampled at anything else is real data (kept in the summary CSV) but must not be
# allowed to silently out-rank a properly-sampled row for the same key just by being
# appended later.
PROTOCOL_WARMUP = 3
PROTOCOL_REPEATS = 15


def _protocol_compliant(row: dict) -> bool:
    return (row.get("warmup") == PROTOCOL_WARMUP and row.get("repeats") == PROTOCOL_REPEATS
            and row.get("measurement_protocol") == PROTOCOL_ID)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, action="append", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--prefix", default="baseline_reference")
    p.add_argument("--run-id", help="only this run may contribute to best_latency")
    args = p.parse_args()
    rows = []
    malformed = 0
    for input_path in args.input:
        for line in input_path.open(encoding="utf-8"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(row, dict) or not row.get("op") or not row.get("case_id"):
                malformed += 1
                continue
            rows.append(row)

    # Keep the last terminal result for a workload.  A resume can contain an early
    # device_fault followed by a successful retry, and the latter is authoritative --
    # *unless* the earlier terminal row was measured under the documented 3-warmup/
    # 15-repeat contract and the later one was not (Finding 4): an exploratory/reduced-
    # sample re-run must not be able to silently outrank a properly-sampled measurement
    # just by being appended later. Two non-compliant rows, or two compliant rows, still
    # resolve the same way as before (last one wins) -- this only blocks the one
    # direction the audit found a real, if currently harmless, occurrence of.
    by_key: dict[tuple, dict] = {}
    protocol_violations = []
    for row in rows:
        key = (row["op"], row["case_id"], row.get("candidate", "manifest_reference"))
        status = row.get("status")
        if key not in by_key:
            by_key[key] = row
            continue
        if status not in TERMINAL:
            continue
        existing = by_key[key]
        if (
            existing.get("status") in TERMINAL
            and _protocol_compliant(existing)
            and not _protocol_compliant(row)
        ):
            protocol_violations.append(
                {
                    "op": row.get("op"),
                    "case_id": row.get("case_id"),
                    "candidate": row.get("candidate", "manifest_reference"),
                    "rejected_status": status,
                    "rejected_warmup": row.get("warmup"),
                    "rejected_repeats": row.get("repeats"),
                    "kept_warmup": existing.get("warmup"),
                    "kept_repeats": existing.get("repeats"),
                }
            )
            continue
        by_key[key] = row

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "op",
        "case_id",
        "family",
        "manifest_status",
        "ref_api",
        "candidate",
        "candidate_source",
        "status",
        "latency_ms",
        "cold_latency_ms",
        "profile_kernel_us",
        "profile_kernel_count",
        "profile_aic_cycles",
        "profile_aiv_cycles",
        "input_mutated",
        "error_type",
        "error",
        "measurement_protocol", "run_id", "checked_outputs", "unchecked_reason",
        "latency_median_ms", "latency_stdev_ms", "latency_min_ms", "latency_max_ms",
    ]
    summary_path = args.output_dir / f"{args.prefix}_profile_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(by_key.values(), key=lambda r: (r["op"], r["case_id"])):
            writer.writerow(row)

    best_path = args.output_dir / f"{args.prefix}_best.csv"
    best_fields = ["op", "case_id", "family", "ref_api", "best_baseline", "best_latency_ms"]
    with best_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=best_fields)
        writer.writeheader()
        groups = defaultdict(list)
        for row in by_key.values():
            if eligible(row) and (not args.run_id or row.get("run_id") == args.run_id):
                groups[(row["op"], row["case_id"])].append(row)
        for (op, case_id), candidates in sorted(groups.items()):
            row = min(candidates, key=lambda r: float(r["latency_ms"]))
            writer.writerow(
                {
                    "op": op,
                    "case_id": case_id,
                    "family": row.get("family"),
                    "ref_api": row.get("ref_api"),
                    "best_baseline": row.get("candidate", "manifest_reference"),
                    "best_latency_ms": row.get("latency_ms"),
                }
            )

    status_path = args.output_dir / f"{args.prefix}_status.json"
    status = Counter(row.get("status") for row in by_key.values())
    status.update({"malformed_lines": malformed})
    status_path.write_text(
        json.dumps(
            {
                "schema": "tileops-ascend950-baseline-summary/v1",
                "input": [str(path) for path in args.input],
                "unique_rows": len(by_key),
                "status_counts": dict(status),
                "measured_rows": sum(status[s] for s in GOOD),
                "best_rows": sum(1 for _ in groups),
                "measurement_protocol": PROTOCOL_ID,
                "run_id": args.run_id,
                "ineligible_measured_rows": sum(r.get("status") in GOOD and not eligible(r) for r in by_key.values()),
                "note": "Only correctness-eligible measured rows enter best_latency; device faults and failures remain visible in summary.",
                # Finding 4: rows that would have silently overwritten a properly-sampled
                # (3-warmup/15-repeat) terminal measurement had this guard not rejected them.
                # Non-empty here is a real signal worth reporting, not something to patch away.
                "protocol_violations_rejected": len(protocol_violations),
                "protocol_violations": protocol_violations,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "unique_rows": len(by_key),
                "status_counts": dict(status),
                "protocol_violations_rejected": len(protocol_violations),
                "summary": str(summary_path),
                "best": str(best_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
