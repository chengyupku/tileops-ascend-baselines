#!/usr/bin/env python3
"""Optional CPU-only progress telemetry for already claimed E2E cases.

Examples (run before the existing guarded device phase):
  python tools/baseline_progress_marker.py --db HOST.sqlite --owner OWNER --batch ID written
  python tools/baseline_progress_marker.py --db HOST.sqlite --owner OWNER --batch ID preflight_passed -- EXISTING_CPU_CHECK ARGS

This is telemetry, not an approval or a baseline prerequisite. A missing marker
means unknown progress. It never changes cases/batches or grants device access.
Only pass an existing, bounded, CPU-only preflight command; NPU work continues
to use the separate host guard. Existing/in-flight cases need no new checks.
"""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time


def connection(path):
    # mode=rw deliberately refuses to create an empty result ledger.
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=rw", uri=True, timeout=3)


def owned_claims(db, batch, owner):
    if not owner or not batch:
        raise ValueError("nonempty exact owner and claimed batch required")
    rows = db.execute("SELECT key,owner,state FROM cases WHERE batch=? LIMIT 4097", (batch,)).fetchall()
    if not rows or len(rows) > 4096:
        raise ValueError("missing or oversized claimed batch")
    if any(actual_owner != owner or state != "claimed" for _, actual_owner, state in rows):
        raise ValueError("all keys must still be claimed by this exact owner")
    keys = []
    for raw, _, _ in rows:
        if not isinstance(raw, str) or len(raw) > 4096:
            raise ValueError("invalid exact case key")
        key = json.loads(raw)
        if (not isinstance(key, list) or len(key) != 3
                or any(not isinstance(v, str) or not v for v in key)):
            raise ValueError("case key must be [library,op,case]")
        keys.append(raw)
    return sorted(keys)


def mark(db_path, owner, batch, stage, command=(), timeout=300):
    if stage not in ("written", "preflight_passed"):
        raise ValueError("unsupported progress stage")
    if not 1 <= timeout <= 300:
        raise ValueError("CPU preflight timeout must be between 1 and 300 seconds")
    command = list(command)
    if command and command[0] == "--":
        command = command[1:]
    if stage == "preflight_passed" and not command:
        raise ValueError("preflight_passed requires an actual existing CPU check command")
    if stage == "written" and command:
        raise ValueError("written does not run a check command")
    with closing(connection(db_path)) as db:
        keys = owned_claims(db, batch, owner)
    started = time.time()
    if stage == "preflight_passed":
        # No SQL transaction spans CPU preparation. Wait for the actual command.
        result = subprocess.run(command, timeout=timeout, check=False)
        if result.returncode != 0:
            return {"marked": False, "stage": stage, "batch": batch,
                    "returncode": result.returncode, "reason": "CPU preflight failed"}
    at = time.time()
    with closing(connection(db_path)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        if owned_claims(db, batch, owner) != keys:
            raise ValueError("claim keys changed during CPU preflight; no stage marked")
        db.execute("CREATE TABLE IF NOT EXISTS case_progress ("
                   "key TEXT PRIMARY KEY, owner TEXT NOT NULL, stage TEXT NOT NULL "
                   "CHECK(stage IN ('written','preflight_passed')), updated_at REAL NOT NULL)")
        # Repeated telemetry cannot regress a successful CPU preflight to written.
        db.executemany(
            "INSERT INTO case_progress (key,owner,stage,updated_at) VALUES (?,?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET owner=excluded.owner, "
            "stage=CASE WHEN case_progress.owner=excluded.owner AND case_progress.stage='preflight_passed' "
            "THEN case_progress.stage ELSE excluded.stage END, updated_at=excluded.updated_at",
            [(key, owner, stage, at) for key in keys])
    return {"marked": True, "stage": stage, "owner": owner, "batch": batch,
            "keys_marked": len(keys), "updated_at": at, "cpu_started_at": started,
            "returncode": 0, "telemetry_only": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--batch", required=True)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("written")
    preflight = sub.add_parser("preflight_passed")
    preflight.add_argument("--timeout", type=int, default=300)
    preflight.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        result = mark(args.db, args.owner, args.batch, args.stage,
                      getattr(args, "command", ()), getattr(args, "timeout", 300))
    except (ValueError, OSError, sqlite3.Error, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"marked": False, "error": str(exc)[:400], "telemetry_only": True}), file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0 if result["marked"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
