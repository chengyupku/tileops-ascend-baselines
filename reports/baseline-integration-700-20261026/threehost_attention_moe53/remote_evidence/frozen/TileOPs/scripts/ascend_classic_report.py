#!/usr/bin/env python3
"""Summarize reviewed full-case baseline coverage without importing NPU runtimes."""
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
DESTINATION = WORKSPACE / "TileOPs/benchmarks/ascend"


def write_csv(name, rows):
    with (DESTINATION / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    matrix = json.loads((DESTINATION / "classic_matrix.json").read_text())
    rows, cases, libraries = matrix["rows"], matrix["cases"], matrix["libraries"]
    expected = {(c["op"], c["case_id"], lib) for c in cases for lib in libraries}
    assert len(rows) == matrix["pairs"] == len(expected)
    assert {(r["op"], r["case_id"], r["library"]) for r in rows} == expected
    for row in rows:
        if row["status"] != "verified":
            continue
        receipt = json.loads((WORKSPACE / row["device_receipt"]).read_text())
        assert all(receipt[k] == row[k] for k in ("op", "case_id", "library"))
        assert receipt["status"] == "verified"
        check = receipt["verification"]
        assert check["checked_outputs"] == check["total_outputs"] > 0
    stamp = datetime.now(timezone.utc).isoformat()
    by_library = []
    for library in libraries:
        counts = Counter(r["status"] for r in rows if r["library"] == library)
        verified, pending, unsupported = (counts[k] for k in (
            "verified", "pending_case_review", "unsupported_within_scope"))
        by_library.append(dict(
            library=library, total_cases=len(cases), verified_cases=verified,
            verified_ratio=f"{verified}/{len(cases)}", pending_case_review=pending,
            unsupported_within_scope=unsupported,
            candidate_or_blocked=len(cases)-verified-pending-unsupported,
            updated_at=stamp))
    by_library.sort(key=lambda r: (-r["verified_cases"], r["library"]))
    write_csv("classic_library_summary.csv", by_library)
    by_operator = []
    for op in sorted({c["op"] for c in cases}):
        selected = [r for r in rows if r["op"] == op]
        verified = [r for r in selected if r["status"] == "verified"]
        total_cases = sum(c["op"] == op for c in cases)
        by_operator.append(dict(
            op=op, total_cases=total_cases,
            cases_with_verified_baseline=len({r["case_id"] for r in verified}),
            verified_case_library_pairs=len(verified), total_case_library_pairs=len(selected),
            verified_pair_ratio=f"{len(verified)}/{len(selected)}",
            pending_case_review=sum(r["status"] == "pending_case_review" for r in selected),
            updated_at=stamp))
    by_operator.sort(key=lambda r: (-r["verified_case_library_pairs"] / r["total_case_library_pairs"], r["op"]))
    write_csv("classic_operator_summary.csv", by_operator)
    print(json.dumps(dict(updated_at=stamp, cases=len(cases), libraries=len(libraries),
                          pairs=len(rows), status_counts=dict(Counter(r["status"] for r in rows)),
                          performance_measured=False)))


if __name__ == "__main__":
    main()
