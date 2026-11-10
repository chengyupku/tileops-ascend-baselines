#!/usr/bin/env python3
"""Snapshot the complete Ascend integration scope without accepting worker claims.

Pure standard library: this command does not import Torch or touch a device.
Reports preserve reported status and link evidence; final acceptance is separate.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_RUN = WORKSPACE / ".agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS"


def flatten(coverage):
    for field in ("rows", "cases"):
        rows = coverage.get(field)
        if isinstance(rows, list) and all(isinstance(row, dict) and "op" in row for row in rows):
            return rows
    rows = []
    for operator in coverage.get("operators", []):
        if not isinstance(operator, dict):
            continue
        for case in operator.get("cases", []):
            rows.append({**{key: value for key, value in operator.items() if key != "cases"},
                         **case, "op": operator["op"]})
    return rows


def snapshot(run):
    inventory_path = run / "inventory.json"
    inventory_bytes = inventory_path.read_bytes()
    inventory = json.loads(inventory_bytes)
    expected = {(case["op"], case["case_id"]) for case in inventory["cases"]}
    if len(expected) != len(inventory["cases"]):
        raise ValueError("Duplicate cases in the authoritative inventory")
    rows, libraries, diagnostics = [], {}, []
    correction_path = run / "oracle_corrections.json"
    corrections = json.loads(correction_path.read_text()).get("issues", []) if correction_path.is_file() else []
    holds = {(issue["op"], case_id): issue for issue in corrections
             for case_id in issue.get("case_ids", [])}
    for token, library in inventory["libraries"].items():
        coverage_path = run / token / "coverage.json"
        if not coverage_path.is_file():
            # Remote deliveries keep their immutable coverage beside the provider.
            coverage_path = (WORKSPACE / "TileOPs/benchmarks/ascend_all_pairs"
                             / token / "coverage.json")
        found = {}
        fingerprint = None
        if coverage_path.is_file():
            try:
                raw = coverage_path.read_bytes()
                coverage = json.loads(raw)
                reported_rows = flatten(coverage)
                declared_library = coverage.get("library")
                if declared_library is None and reported_rows and all(
                    item.get("library") == library for item in reported_rows
                ):
                    declared_library = library
                if declared_library != library:
                    raise ValueError(f"Unexpected library {coverage.get('library')!r}")
                fingerprint = hashlib.sha256(raw).hexdigest()
                for item in reported_rows:
                    if item.get("library", library) != library:
                        raise ValueError(f"Wrong library in row: {item.get('library')!r}")
                    key = (item["op"], item["case_id"])
                    if key not in expected or key in found:
                        raise ValueError(f"Unknown or duplicate case {key}")
                    found[key] = item
            except (OSError, ValueError, KeyError, TypeError) as exc:
                diagnostics.append({"library": library, "error": str(exc),
                                    "file": str(coverage_path)})
                found = {}
        counts = Counter()
        for op, case_id in sorted(expected):
            item = found.get((op, case_id), {})
            status = item.get("status", "pending_worker_report")
            counts[status] += 1
            rows.append({
                "op": op, "case_id": case_id, "library": library,
                "reported_status": status,
                "deployment_status": item.get("deployment_status"),
                "reference_blocked": op in inventory.get("reference_blocked_ops", []),
                "accepted_by_this_report": False,
                "reason": item.get("reason"),
                "receipt": item.get("receipt") or item.get("device_receipt"),
                "worker_acceptance": item.get("acceptance") or item.get("root_review"),
                "oracle_review_required": holds.get((op, case_id)),
                "coverage_file": str(coverage_path) if fingerprint else None,
                "coverage_sha256": fingerprint,
            })
        libraries[library] = {"reported_cases": len(found), "expected_cases": len(expected),
                              "reported_status_counts": dict(counts)}
    return {
        "schema": "ascend-all-pairs-progress/v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {"operators": len(inventory["operators"]), "libraries": len(libraries),
                  "cases": len(expected), "case_library_pairs": len(rows)},
        "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "meaning": "Worker reports only. This snapshot does not grant numerical/native acceptance or a performance ranking.",
        "missing_factory_ops": inventory["missing_factory_ops"],
        "reference_blocked_ops": inventory.get("reference_blocked_ops", []),
        "oracle_corrections_file": str(correction_path) if corrections else None,
        "libraries": libraries, "diagnostics": diagnostics, "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = snapshot(args.run)
    output = args.output or args.run / "coverage_snapshot.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps({"scope": result["scope"], "libraries": result["libraries"],
                      "diagnostics": result["diagnostics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
