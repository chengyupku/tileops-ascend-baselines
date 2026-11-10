"""Run the shared benchmark on the dedicated remote; export auditable evidence."""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ascend950_snapshot import PROTOCOL, PUBLISHER, PUBLISHER_REPOSITORY, seal, validate

EVIDENCE_FILES = (
    "receipt.json", "result.json", "environment.json", "results.jsonl", "sources.json",
)


def export_run(source, target, performance=False):
    target.mkdir()
    for name in (*EVIDENCE_FILES, *(("clock-readbacks.json",) if performance else ())):
        shutil.copy2(source / name, target / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tileops", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.tileops.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "preflight.json").write_text(json.dumps({
        "status": "started", "source": str(root),
        "config": str(args.config), "hardware_started": False,
    }, indent=2) + "\n")
    # Standard-library preflight, before torch/ACL imports or device allocation.
    config = json.loads(args.config.read_text())
    if config.get("host_reservation_confirmed") is not True:
        raise RuntimeError("Host-shared lock, PID visibility and physical device mapping are not confirmed")
    guard = Path(config["guard"]).resolve(strict=True)
    clock = Path(config["clock_config"]).resolve(strict=True)
    if not Path("/run/lock/ascend-npu/device-0.lock").is_file():
        raise RuntimeError("Host reservation lock is not mounted")
    env = dict(os.environ, TILEOPS_NPU_GUARD=str(guard))
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True):
        raise RuntimeError("The measured source checkout is dirty")
    run_id = os.environ["GITHUB_RUN_ID"]
    attempt = os.environ["GITHUB_RUN_ATTEMPT"]
    if os.environ.get("GITHUB_REPOSITORY") != "yyttt6/TileOPs.github.io":
        raise RuntimeError("Only the website repository's dedicated workflow can publish")
    if os.environ.get("GITHUB_REF") != "refs/heads/main":
        raise RuntimeError("Official CI runs only from the main workflow")
    base = [sys.executable, str(root / "scripts/benchmark_ascend.py"),
            "--owner", f"remote-nightly/{run_id}/{attempt}", "--guard", str(guard),
            "--wait", "600", "--timeout", "10800", "--op", "all"]
    subprocess.run([*base, "--run-dir", str(output / "correctness"), "--verify-only"],
                   cwd=root, env=env, check=True)
    factories = config.get("implementation_factories", [])
    provenance = config.get("baseline_provenance", {})
    if bool(factories) != bool(provenance):
        raise ValueError("Baseline factories and their provenance must be configured together")
    if factories:
        command = [*base, "--run-dir", str(output / "performance"), "--clock-config", str(clock)]
        for factory in factories:
            command.extend(["--implementation-factory", factory])
        subprocess.run(command, cwd=root, env=env, check=True)
    snapshot = output / "snapshot"
    snapshot.mkdir()
    exported_provenance = {}
    for tag, entry in provenance.items():
        source = Path(entry["source_path"])
        relative = f"baseline-sources/{tag}{source.suffix}"
        target = snapshot / relative
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
        exported_provenance[tag] = {k: v for k, v in entry.items() if k != "source_path"}
        exported_provenance[tag]["evidence_path"] = relative
    export_run(output / "correctness", snapshot / "correctness")
    correct = json.loads((output / "correctness/result.json").read_text())
    environment = json.loads((output / "correctness/environment.json").read_text())
    count = 0
    if factories:
        export_run(output / "performance", snapshot / "performance", performance=True)
        rows = [json.loads(s) for s in (output / "performance/results.jsonl").read_text().splitlines()]
        count = sum(len(r["implementations"]) > 1 for r in rows)
        if not count:
            raise RuntimeError("Configured baselines produced no matched comparisons")
    shutil.copytree(root / "src/tileops/manifest", snapshot / "source/src/tileops/manifest",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    meta = {
        "schema_version": 1, "status": "passed", "measurement_protocol": PROTOCOL,
        "publisher": PUBLISHER, "publisher_repository": PUBLISHER_REPOSITORY,
        "commit": commit, "run_id": run_id, "run_attempt": attempt,
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gpu": environment["device"], "environment": {
            **environment,
            "cann": environment.get("cann_components", {}).get("CANN"),
            "driver": environment.get("cann_components", {}).get("DRIVER"),
        },
        "correctness_count": correct["passed"], "comparison_count": count,
        "baseline_provenance": exported_provenance,
        "clock_verified": bool(count), "guard_verified": True,
    }
    (snapshot / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    seal(snapshot)
    validate(snapshot)
    print(json.dumps({"commit": commit, "correctness": correct["passed"], "comparisons": count}))


if __name__ == "__main__":
    main()
