"""Run frozen main-source operators sequentially, with one guard reservation per operator."""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ascend950_snapshot import seal
from nightly_snapshot import PROTOCOL, build_cache, validate

HERE = Path(__file__).resolve().parent


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temp.replace(path)


def identity(commit):
    files = [HERE / "nightly_device.py", *sorted((HERE / "nightly_support").rglob("*.py"))]
    return dict(
        commit=commit,
        measurement_files={
            str(f.relative_to(HERE)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files
        },
        toolchain=hashlib.sha256(
            Path("/opt/tileops-ci/server2-20261030/config/toolchain-v1.lock.json").read_bytes()
        ).hexdigest(),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--operators", default="all")
    p.add_argument("--resume", type=Path)
    a = p.parse_args()
    a.source = a.source.resolve()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    assert os.environ["GITHUB_REPOSITORY"] == "yyttt6/TileOPs.github.io"
    assert os.environ["GITHUB_REF"] == "refs/heads/main"
    commit = subprocess.check_output(
        ["git", "-C", str(a.source), "rev-parse", "HEAD"], text=True
    ).strip()
    assert not subprocess.check_output(
        ["git", "-C", str(a.source), "status", "--porcelain"], text=True
    ).strip()
    current_identity = identity(commit)
    write(out / "identity.json", current_identity)
    prior = {}
    if a.resume:
        a.resume = a.resume.resolve()
        assert json.loads((a.resume / "identity.json").read_text()) == current_identity, (
            "Checkpoint environment mismatch"
        )
        prior = json.loads((a.resume / "receipts.json").read_text())
    start = time.monotonic()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    timings = []

    def run(op, target, attempt=0):
        command = [
            "npu-guard",
            "run",
            "--owner",
            f"nightly/{os.environ['GITHUB_RUN_ID']}/{op}/{attempt}",
            "--wait",
            "1800",
            "--timeout",
            "7200",
            "--",
            sys.executable,
            str(HERE / "nightly_device.py"),
            "--source",
            str(a.source),
            "--output",
            str(target),
            "--op",
            op,
        ]
        t = time.monotonic()
        with (out / f"{op}-attempt{attempt}.log").open("x") as log:
            r = subprocess.run(command, cwd=a.source, stdout=log, stderr=subprocess.STDOUT)
        timings.append(
            dict(
                op=op,
                attempt=attempt,
                guard_exit_code=r.returncode,
                elapsed_seconds=time.monotonic() - t,
            )
        )
        write(
            out / "timing.json",
            dict(started_at=started, elapsed_seconds=time.monotonic() - start, stages=timings),
        )
        return r.returncode

    assert run("__list__", out / "selection.json") == 0
    selection = json.loads((out / "selection.json").read_text())
    ops = selection["operators"] if a.operators == "all" else a.operators.split(",")
    assert set(ops) <= set(selection["operators"]) and len(ops) == len(set(ops))
    write(out / "plan.json", dict(commit=commit, operators=ops, started_at=started))
    sys.path.insert(0, str(HERE / "nightly_support"))
    from complete_profile import complete_digest, decode, pure_aicpu_base
    from tools.measure_baselines import _parse_ffts_profile

    receipts = {}
    measured = []
    failures = []
    frozen_sources = None
    for op in ops:
        saved = prior.get(op, {})
        if saved.get("guard_exit_code") == 0 and saved.get("result", {}).get("status") == "passed":
            assert saved["hardware"] == selection["hardware"], "Checkpoint hardware mismatch"
            assert not saved["result"]["failed"] and not saved["result"]["source_changed"]
            directory = saved.get("attempts", [{}])[saved.get("selected_attempt", 0)].get(
                "directory", op
            )
            origin = (a.resume / directory).resolve()
            assert origin.is_relative_to(a.resume)
            target = out / op
            shutil.copytree(origin, target)
            code, result = 0, saved["result"]
            receipts[op] = dict(
                saved,
                resumed_from=str(a.resume),
                attempts=[dict(attempt=0, directory=op)],
                selected_attempt=0,
            )
            print("RESUME", op, "CPU replay of retained raw profiles", flush=True)
        else:
            attempts = []
            for attempt in range(2):
                target = out / (op if attempt == 0 else f"{op}-retry{attempt}")
                code = run(op, target, attempt)
                result = (
                    json.loads((target / "result.json").read_text())
                    if (target / "result.json").exists()
                    else {}
                )
                receipt = dict(
                    guard_exit_code=code,
                    result=result,
                    hardware=json.loads((target / "hardware.json").read_text())
                    if (target / "hardware.json").exists()
                    else {},
                )
                attempts.append(dict(attempt=attempt, directory=target.name, **receipt))
                if code not in (75, 76, 78) or result.get("failed", 0):
                    break
            receipts[op] = dict(**receipt, attempts=attempts, selected_attempt=attempt)
        if code or result.get("status") != "passed":
            failures.append(op)
        else:
            sources = json.loads((target / "sources.json").read_text())
            if frozen_sources is None:
                frozen_sources = sources
            assert sources == frozen_sources
            case_rows = [
                json.loads(line) for line in (target / "results.jsonl").read_text().splitlines()
            ]
            assert {r["case_id"] for r in case_rows} == set(selection["cases"][op])
            assert len(case_rows) == len(selection["cases"][op])
            for line in (target / "results.jsonl").read_text().splitlines():
                row = json.loads(line)
                coverage = json.loads(
                    Path(row["measurement"]["profile_coverage_evidence"]).read_text()
                )
                for c in coverage[-15:]:
                    assert complete_digest(c["path"]) == c["complete_sha256"]
                    extra = decode(c["path"])
                    try:
                        replay = _parse_ffts_profile(Path(c["path"]), 1)
                    except ValueError as exc:
                        if str(exc) != "No valid FFTS activity":
                            raise
                        replay = pure_aicpu_base(extra)
                    import math

                    assert math.isclose(
                        replay["latency_ms"] + extra["added_us"] / 1000,
                        c["result"]["latency_ms"],
                        rel_tol=1e-10,
                    )
                row["raw_replayed"] = True
                row["measurement"] = {
                    k: v
                    for k, v in row["measurement"].items()
                    if k
                    not in (
                        "profile_coverage_adapter",
                        "profile_coverage_evidence",
                        "retained_profiles",
                    )
                }
                row["coverage"] = [
                    {"complete_sha256": c["complete_sha256"], "result": c["result"]}
                    for c in coverage
                ]
                row["workload"] = dict(
                    op=op,
                    case_id=row["case_id"],
                    family=row["family"],
                    params=row["params"],
                    tensors={
                        k: v
                        for k, v in row["params"].items()
                        if isinstance(v, list) and len(v) == 2 and isinstance(v[0], list)
                    },
                )
                measured.append(row)
        write(out / "receipts.json", receipts)
        print(op, result.get("passed"), result.get("failed"), "guard", code, flush=True)
    assert not failures, f"Failed operators; no publication: {failures}"
    snapshot = out / "snapshot"
    snapshot.mkdir()
    base = json.loads((HERE.parent / "data/baseline-cache/snapshot.json").read_text())
    # Bind unpaired workloads to their family from the frozen source manifest.
    for row in measured:
        row["workload"]["family"] = row["family"]
    date = datetime.datetime.now(datetime.timezone.utc).isoformat()
    meta = dict(
        schema_version=2,
        publisher="server2-physical32",
        publisher_repository="https://github.com/yyttt6/TileOPs.github.io",
        status="passed",
        measurement_protocol=PROTOCOL,
        commit=commit,
        date=date,
        operators=ops,
        case_ids={op: selection["cases"][op] for op in ops},
        fresh_cases=len(measured),
        run_id=os.environ["GITHUB_RUN_ID"],
        run_attempt=os.environ["GITHUB_RUN_ATTEMPT"],
        elapsed_seconds=time.monotonic() - start,
    )
    for name, value in [
        ("meta", meta),
        ("sources", frozen_sources),
        ("measurements", measured),
        ("receipts", receipts),
        ("baseline-before", base),
        ("baseline-cache", build_cache(base, measured, commit, date)),
    ]:
        write(snapshot / f"{name}.json", value)
    seal(snapshot)
    validate(snapshot)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
