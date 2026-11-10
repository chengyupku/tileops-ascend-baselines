#!/usr/bin/env python3
"""Refresh the first-case pilot summaries using reviewed receipts; CPU-only."""

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
    matrix = json.loads((DESTINATION / "pilot_matrix.json").read_text())
    rows = matrix["rows"]
    keys = [(r["op"], r["case_id"], r["library"]) for r in rows]
    assert len(keys) == len(set(keys)) == matrix["pairs"]
    operators = sorted({r["op"] for r in rows})
    libraries = sorted({r["library"] for r in rows})
    assert len(rows) == len(operators) * len(libraries)
    for op in operators:
        assert len({r["case_id"] for r in rows if r["op"] == op}) == 1
    stamp = datetime.now(timezone.utc).isoformat()
    jobs = []
    for row in rows:
        if row["status"] != "verified":
            continue
        receipt = WORKSPACE / row["device_receipt"]
        result = json.loads(receipt.read_text())
        assert all(result[k] == row[k] for k in ("op", "case_id", "library"))
        assert result["status"] == "verified"
        check = result["verification"]
        assert check["checked_outputs"] == check["total_outputs"] > 0
        original_jobs = json.loads(receipt.with_name("jobs.json").read_text())
        matches = [j for j in original_jobs
                   if all(j.get(k) == row[k] for k in ("op", "case_id", "library"))]
        assert len(matches) == 1, (receipt, matches)
        # Preserve the actual successful environment, including vendor selection.
        jobs.append({**matches[0], "profile": False})

    by_library = []
    for library in libraries:
        selected = [r for r in rows if r["library"] == library]
        counts = Counter(r["status"] for r in selected)
        verified = counts["verified"]
        unsupported = counts["unsupported_within_scope"]
        deployment = sum(n for s, n in counts.items() if "deployment" in s)
        awaiting = sum(counts[s] for s in (
            "pending_root_review", "adapter_written_unverified", "adapter_written_unregistered"))
        by_library.append(dict(
            library=library, representative_cases=len(selected), verified=verified,
            verified_ratio=f"{verified}/{len(selected)}", adapters_awaiting_test=awaiting,
            deployment_blocked=deployment,
            other_blocked_or_failed=len(selected)-verified-unsupported-deployment-awaiting,
            unsupported_within_scope=unsupported, snapshot_status="phase1_snapshot",
            updated_at=stamp))
    by_library.sort(key=lambda r: (-r["verified"], r["library"]))
    write_csv("pilot_summary.csv", by_library)

    by_operator = []
    for op in operators:
        selected = [r for r in rows if r["op"] == op]
        verified = sorted(r["library"] for r in selected if r["status"] == "verified")
        unsupported = sum(r["status"] == "unsupported_within_scope" for r in selected)
        by_operator.append(dict(
            op=op, case_id=selected[0]["case_id"], verified_libraries=len(verified),
            total_libraries=len(selected), verified_ratio=f"{len(verified)}/{len(selected)}",
            verified_library_names=";".join(verified), unsupported_within_scope=unsupported,
            other_states=len(selected)-len(verified)-unsupported, updated_at=stamp))
    by_operator.sort(key=lambda r: (-r["verified_libraries"], r["op"]))
    write_csv("pilot_operator_summary.csv", by_operator)
    jobs.sort(key=lambda j: (j["op"], j["library"]))
    (DESTINATION / "pilot_verified_jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
    summary = dict(
        updated_at=stamp, scope="phase_1_one_frozen_case_per_operator",
        operators=len(operators), libraries=len(libraries), pairs=len(rows),
        statuses=dict(Counter(r["status"] for r in rows)),
        verified_receipts_checked=len(jobs), performance_measured=False,
        note="Passing one case does not establish support for other cases. "
             "Job list preserves each passing receipt's environment and excludes quarantines.")
    (DESTINATION / "pilot_readiness.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
