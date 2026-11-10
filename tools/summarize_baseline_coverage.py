#!/usr/bin/env python3
"""Write manifest-op coverage alongside baseline profile results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    sys.path.insert(0, str(root / "TileOPs" / "src"))
    sys.path.insert(0, str(root / "TileOPs"))
    from benchmarks._cases import registered
    from benchmarks.api import cases
    from tileops.manifest import load_manifest

    manifest = load_manifest()
    registered_ops = set(registered())
    by_op: dict[str, dict] = defaultdict(dict)
    malformed = 0
    for line in args.profile.open(encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if not isinstance(row, dict) or not row.get("op") or not row.get("case_id"):
            malformed += 1
            continue
        key = (row["op"], row["case_id"], row.get("candidate", "manifest_reference"))
        by_op.setdefault(row["op"], {})[key] = row

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "baseline_reference_coverage.csv"
    fields = [
        "op",
        "family",
        "manifest_status",
        "case_factory",
        "case_count",
        "profile_rows",
        "measured_rows",
        "failed_rows",
        "device_fault_rows",
        "coverage_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for op in sorted(manifest):
            entry = manifest[op]
            has_factory = op in registered_ops
            case_count = 0
            if has_factory:
                try:
                    case_count = len(cases(op))
                except Exception:
                    has_factory = False
            rows = [r for key, r in by_op.get(op, {}).items() if key[2] == "manifest_reference"]
            counts = Counter(r.get("status") for r in rows)
            measured = sum(v for k, v in counts.items() if str(k).startswith("measured"))
            if not has_factory:
                coverage = "no_case_factory"
            elif measured == case_count and case_count:
                coverage = "complete"
            elif measured or rows:
                coverage = "partial"
            else:
                coverage = "unmeasured"
            writer.writerow(
                {
                    "op": op,
                    "family": entry.get("family"),
                    "manifest_status": entry.get("status"),
                    "case_factory": has_factory,
                    "case_count": case_count,
                    "profile_rows": len(rows),
                    "measured_rows": measured,
                    "failed_rows": counts.get("failed", 0),
                    "device_fault_rows": counts.get("device_fault", 0),
                    "coverage_status": coverage,
                }
            )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
