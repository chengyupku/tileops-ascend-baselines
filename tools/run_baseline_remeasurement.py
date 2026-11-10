#!/usr/bin/env python3
"""Serial, journaled full rerun. CPU supervisor; every NPU worker is owned by npu-guard.

No daemonization or escaped children. Keep this command in a managed foreground session.
Timeouts terminate the guard, which drains/terminates its own process group before this
supervisor records a terminal outcome and resumes the remaining cases in a fresh process.
"""
from __future__ import annotations

import argparse
import collections
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.baseline_protocol import PROTOCOL_ID, eligible, provenance


def read_rows(path):
    if not path.exists():
        return []
    rows = []
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed JSONL: {path}:{number}; preserve and inspect") from exc
    return rows


def append(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(provenance(row), ensure_ascii=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def atomic(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temp.replace(path)


def key(row):
    return row.get("op"), row.get("case_id"), row.get("candidate")


def check_sources(run):
    sources = json.loads((run / "sources.json").read_text())
    for name, digest in sources["sha256"].items():
        path = ROOT / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen source changed: {name}; stopping before mixing versions")


def summarize(run, plan, complete=False):
    refs = read_rows(run / "reference.jsonl")
    candidates = read_rows(run / "candidates.jsonl")
    own = read_rows(run / "tileops.jsonl")
    latest = {key(r): r for r in refs + candidates}
    best = {}
    for row in latest.values():
        if not eligible(row) or row.get("run_id") != plan["run_id"]:
            continue
        pair = row["op"], row["case_id"]
        if pair not in best or row["latency_ms"] < best[pair]["latency_ms"]:
            best[pair] = row
    with (run / "best.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["op", "case_id", "best_baseline", "best_latency_ms"])
        writer.writeheader()
        for pair, row in sorted(best.items()):
            writer.writerow(dict(op=pair[0], case_id=pair[1], best_baseline=row["candidate"], best_latency_ms=row["latency_ms"]))
    with (run / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["op", "case_id", "status", "baseline_ms", "kernel_ms", "baseline_over_kernel", "passes_80pct"])
        writer.writeheader()
        for row in {key(r): r for r in own}.values():
            pair = row["op"], row["case_id"]
            out = dict(op=pair[0], case_id=pair[1], status=row.get("status"))
            if pair in best and eligible(row):
                ratio = best[pair]["latency_ms"] / row["latency_ms"]
                out.update(baseline_ms=best[pair]["latency_ms"], kernel_ms=row["latency_ms"], baseline_over_kernel=ratio, passes_80pct=ratio >= .8)
            writer.writerow(out)
    covered = {(r["op"], r["case_id"]) for r in refs}
    inventoried = {(r["op"], r["case_id"]) for r in candidates if r.get("candidate") == "__discovery__"}
    expected = {(r["op"], r["case_id"]) for r in plan["expected_cases"]}
    own_expected = {(op, case_id) for op, spec in plan["own_ops"].items() for case_id in spec["cases"]}
    own_seen = {(r["op"], r["case_id"]) for r in own}
    inventory_missing = []
    for row in candidates:
        if row.get("status") == "candidate_inventory":
            for candidate in row.get("candidates", []):
                if (row["op"], row["case_id"], candidate) not in latest:
                    inventory_missing.append([row["op"], row["case_id"], candidate])
    report = {
        "run_id": plan["run_id"], "measurement_protocol": PROTOCOL_ID,
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "execution_finished": complete, "reference_cases_recorded": len(covered), "expected_cases": len(expected),
        "candidate_cases_inventoried": len(inventoried), "best_baseline_cases": len(best),
        "reference_missing": sorted(expected - covered), "candidate_inventory_missing": sorted(expected - inventoried),
        "candidate_results_missing": inventory_missing,
        "tileops_cases_recorded": len({key(r) for r in own}),
        "tileops_cases_missing": sorted(own_expected - own_seen),
        "historical_candidates_missing": [k for k in plan.get("historical_candidate_keys", []) if tuple(k) not in latest],
        "status_counts": dict(collections.Counter(r.get("status") for r in latest.values())),
        "no_case_factory_ops": plan["no_case_factory_ops"],
        "note": "Partial until execution_finished and coverage reconciled. Failures/timeouts are not performance results.",
    }
    atomic(run / "coverage.json", report)
    return report


def run_job(run, plan, phase, op, job_index):
    check_sources(run)
    ops = [op] if isinstance(op, str) else list(op)
    op = ",".join(ops)
    output = run / {"reference": "reference.jsonl", "candidates": "candidates.jsonl", "tileops": "tileops.jsonl"}[phase]
    progress = run / "active.json"
    expected = [r for r in plan["expected_cases"] if r["op"] in ops]
    if phase == "tileops":
        expected = [r for r in expected if r["case_id"] in plan["own_ops"][r["op"]]["cases"]]
    owner = f"root/remeasure-v2/{phase}/{job_index:03d}"
    maximum_attempts = len(expected) * 8 + 8
    for attempt in range(maximum_attempts):
        previous = read_rows(output)
        start_count = len(previous)
        progress.unlink(missing_ok=True)
        script = {"reference": "measure_baselines.py", "candidates": "measure_baseline_candidates.py", "tileops": "measure_tileops_ascend.py"}[phase]
        argv = [str(ROOT / ".venv/bin/python"), str(ROOT / "tools" / script), "--output", str(output), "--warmup", "3", "--repeats", "15"]
        for name in ops:
            argv += ["--op", name]
        if phase == "candidates":
            # Preserve all benchmark-declared opponents and every historically successful
            # compile opponent. Only NEW automatic compile attempts follow the no-baseline rule.
            successful_refs = {(r["op"], r["case_id"]) for r in read_rows(run / "reference.jsonl") if eligible(r)}
            allow = set(plan["previously_measured_compile"])
            allow.update(r["op"] + "::" + r["case_id"] for r in expected if (r["op"], r["case_id"]) not in successful_refs)
            allow_path = run / "fallback_allow.txt"
            allow_path.write_text("\n".join(sorted(allow)) + "\n")
            argv += ["--fallback-allow-file", str(allow_path), "--sync-every", "1"]
        if phase == "tileops":
            argv += ["--cases-file", str(run / "own_cases.txt"), "--sync-every", "1"]
            for module in sorted({m for name in ops for m in plan["own_ops"][name]["modules"]}):
                argv += ["--register-module", module]
        shell = "source " + shlex.quote(str(ROOT / "env.sh")) + "\nexec " + shlex.join(argv)
        command = [sys.executable, str(ROOT / "tools/npu_guard.py"), "run", "--owner", owner,
                   "--wait", "600", "--timeout", "21600", "--", "bash", "-c", shell]
        env = os.environ.copy()
        env.update(BASELINE_PROGRESS_FILE=str(progress), PYTHONUNBUFFERED="1",
                   PYTHONPATH=f"{ROOT}:{ROOT / 'TileOPs/src'}:{ROOT / 'TileOPs'}",
                   OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
                   NUMEXPR_NUM_THREADS="8", TORCHINDUCTOR_COMPILE_THREADS="4")
        log = run / "logs" / f"{job_index:04d}-{phase}-attempt{attempt:03d}.log"
        active = None
        timed_out = False
        interrupted = False
        launched = time.time()
        print(f"START {job_index} {phase} {op} attempt={attempt}", flush=True)
        with log.open("w") as stream:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            def stop(_sig, _frame):
                nonlocal interrupted
                interrupted = True
                proc.terminate()  # only our guard; it cleans up its entire workload group
            old_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
            try:
                while proc.poll() is None:
                    if active is None and time.time() - launched > 1200 and not timed_out:
                        timed_out = True
                        proc.terminate()
                    if progress.exists():
                        active = json.loads(progress.read_text())
                        budget = plan["discovery_timeout_s"] if active["stage"] == "discovery" else plan[{"reference": "reference_timeout_s", "candidates": "candidate_timeout_s", "tileops": "kernel_timeout_s"}[phase]]
                        if time.time() - active["started_at"] > budget and not timed_out:
                            timed_out = True
                            proc.terminate()
                    atomic(run / "status.json", {"state": "running", "job": job_index, "phase": phase, "op": op,
                           "attempt": attempt, "supervisor_pid": os.getpid(), "guard_pid": proc.pid,
                           "owner": owner, "active": active, "log": str(log), "updated_at": time.time()})
                    time.sleep(2)
                rc = proc.wait()
            finally:
                for sig, handler in old_handlers.items():
                    signal.signal(sig, handler)
        if interrupted:
            raise InterruptedError("User/session stopped supervisor; worker cleaned up by guard")
        new_rows = read_rows(output)[start_count:]
        append(run / "events.jsonl", {"job": job_index, "phase": phase, "op": op, "attempt": attempt,
               "returncode": rc, "timed_out": timed_out, "new_rows": len(new_rows), "log": str(log)})
        if rc == 76:
            for row in new_rows:
                if str(row.get("status", "")).startswith("measured"):
                    append(output, {**row, "status": "invalidated_external_activity", "error": "npu-guard detected external activity"})
            raise RuntimeError("External NPU activity; attempt invalidated and run stopped")
        if rc in (75, 78):
            raise RuntimeError(f"npu-guard refused/aborted with {rc}; no bypass or automatic device reset")
        if timed_out or (rc not in (0, 2, 3) and active):
            if active is None:
                raise RuntimeError(f"Worker startup/discovery exceeded 1200 seconds: {log}")
            row = active["row"]
            if not any(key(r) == key(row) for r in new_rows):
                append(output, {**row, "status": "timeout" if timed_out else "interrupted",
                       "error_type": "CaseDeadline" if timed_out else "WorkerExit",
                       "error": f"stage={active['stage']}; guard_exit={rc}; see {log.name}",
                       "elapsed_wall_s": time.time() - active["started_at"]})
            elif not new_rows:
                raise RuntimeError(f"Worker made no progress: {log}")
            continue
        if rc == 3:
            if not new_rows:
                raise RuntimeError(f"Device fault without persisted result: {log}")
            continue
        if rc not in (0, 2):
            raise RuntimeError(f"Worker exited before a case was journaled: rc={rc}; {log}")
        # A clean CLI exit is not coverage: reconcile every expected workload explicitly.
        rows = read_rows(output)
        seen = {(r["op"], r["case_id"]) for r in rows if r["op"] in ops and (phase != "candidates" or r.get("candidate") == "__discovery__")}
        missing = [r for r in expected if (r["op"], r["case_id"]) not in seen]
        if missing:
            raise RuntimeError(f"{phase}/{op}: {len(missing)} expected cases silently missing; see {log}")
        if phase == "candidates":
            actual = {key(r) for r in rows}
            inventories = {(r["op"], r["case_id"]): r for r in rows if r.get("candidate") == "__discovery__"}
            for old_key in plan.get("historical_candidate_keys", []):
                name, case_id, candidate = old_key
                if name in ops and tuple(old_key) not in actual:
                    inventory = inventories.get((name, case_id), {})
                    append(output, {"op": name, "case_id": case_id, "candidate": candidate,
                           "warmup": 3, "repeats": 15, "status": "no_implementation",
                           "error": "Previously measured candidate not exposed by current factory discovery",
                           "factory_errors": inventory.get("factory_errors"),
                           "discovery_status": inventory.get("status")})
        return
    raise RuntimeError(f"Too many restarts for {phase}/{op}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    plan = json.loads((run / "plan.json").read_text())
    os.environ["BASELINE_RUN_ID"] = plan["run_id"]
    os.environ["BASELINE_TOOLCHAIN_SHA256"] = hashlib.sha256((run / "sources.json").read_bytes()).hexdigest()
    lock = (run / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # single journal writer, not an NPU lock
    (run / "logs").mkdir(exist_ok=True)
    for name in ("reference.jsonl", "candidates.jsonl", "tileops.jsonl", "events.jsonl"):
        (run / name).touch(exist_ok=True)
    completed = {r["job"] for r in read_rows(run / "events.jsonl") if r.get("status") == "job_complete"}
    # Secure a reference result for the full manifest before potentially slow compiler work.
    jobs = [("reference", plan["ops"][i:i+12]) for i in range(0, len(plan["ops"]), 12)]
    for i in range(0, len(plan["ops"]), 8):
        chunk = plan["ops"][i:i+8]
        jobs.append(("candidates", chunk))
        own_chunk = [op for op in chunk if op in plan["own_ops"]]
        if own_chunk:
            jobs.append(("tileops", own_chunk))
    try:
        for index, (phase, op) in enumerate(jobs):
            if index in completed:
                continue
            run_job(run, plan, phase, op, index)
            append(run / "events.jsonl", {"job": index, "phase": phase, "op": op, "status": "job_complete"})
            report = summarize(run, plan)
            print(f"DONE {index + 1}/{len(jobs)} reference={report['reference_cases_recorded']}/{report['expected_cases']} best={report['best_baseline_cases']}", flush=True)
        report = summarize(run, plan, complete=True)
        if (report["reference_missing"] or report["candidate_inventory_missing"] or report["candidate_results_missing"]
                or report["tileops_cases_missing"] or report["historical_candidates_missing"]):
            raise RuntimeError("Coverage reconciliation failed; see coverage.json")
        atomic(run / "status.json", {"state": "completed", "supervisor_pid": os.getpid(), "updated_at": time.time(), "coverage": report})
        print("FULL RERUN FINISHED; see coverage.json, best.csv, comparison.csv", flush=True)
    except BaseException as exc:
        summarize(run, plan)
        atomic(run / "status.json", {"state": "stopped", "supervisor_pid": os.getpid(), "error": repr(exc), "updated_at": time.time()})
        raise


if __name__ == "__main__":
    main()
