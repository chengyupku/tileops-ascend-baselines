#!/usr/bin/env python3
"""Run chunked, npu-guarded profiling processes over the manifest.

An Ascend device fault poisons the current Python process.  This launcher therefore
does not run all 191 case factories in one process.  A chunk normally contains several
ops to amortize profiler startup; a device fault restarts only that chunk in a fresh
process and reservation.  Successful rows are skipped by the JSONL writer, so the
launcher is safe to resume after a fault or a host interruption.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "tools" / "npu_guard.py"
MEASURE = ROOT / "tools" / "measure_baselines.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--op", action="append", help="restrict to one or more ops")
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--max-device-restarts", type=int, default=32)
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="maximum wall time in seconds for one guarded chunk",
    )
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")

    # This import is CPU-only; NPU initialization happens only inside each guarded
    # child command below.
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    sys.path.insert(0, str(ROOT / "TileOPs" / "src"))
    sys.path.insert(0, str(ROOT / "TileOPs"))
    from benchmarks._cases import registered
    from tileops.manifest import load_manifest

    manifest = load_manifest()
    names = sorted(set(manifest) & set(registered()))
    if args.op:
        names = [name for name in names if name in set(args.op)]
    if args.op:
        wanted = set(args.op)
        names = [name for name in names if name in wanted]
    print(f"profiling {len(names)} ops into {args.output}", flush=True)

    failures = 0
    chunks = [names[i : i + args.chunk_size] for i in range(0, len(names), args.chunk_size)]
    for chunk_index, chunk in enumerate(chunks):
        owner = f"root/baseline-profile/chunk-{chunk_index:03d}"
        op_args = " ".join(f"--op {shlex.quote(op)}" for op in chunk)
        child = (
            f"source {shlex.quote(str(ROOT / 'env.sh'))}; "
            f"export PYTHONPATH={shlex.quote(str(ROOT / 'TileOPs' / 'src'))}:"
            f"{shlex.quote(str(ROOT / 'TileOPs'))}:$PYTHONPATH; "
            f"exec {shlex.quote(sys.executable)} {shlex.quote(str(MEASURE))} "
            f"{op_args} --warmup {args.warmup} --repeats {args.repeats} "
            f"--output {shlex.quote(str(args.output))}"
        )
        print(
            f"===== chunk {chunk_index + 1}/{len(chunks)}: {', '.join(chunk)} ({owner}) =====",
            flush=True,
        )
        restarts = 0
        visibility_retries = 0
        while True:
            result = subprocess.run(
                [
                    sys.executable,
                    str(GUARD),
                    "run",
                    "--owner",
                    owner,
                    "--wait",
                    "600",
                    "--timeout",
                    str(args.timeout),
                    "--",
                    "bash",
                    "-lc",
                    child,
                ],
                cwd=ROOT,
            )
            if result.returncode == 78:
                visibility_retries += 1
                if visibility_retries > args.max_device_restarts:
                    print(
                        f"device visibility did not recover while measuring chunk {chunk_index}",
                        flush=True,
                    )
                    return 78
                print(
                    f"device visibility unavailable for chunk {chunk_index}; waiting before retry "
                    f"({visibility_retries}/{args.max_device_restarts})",
                    flush=True,
                )
                time.sleep(5)
                continue
            if result.returncode != 3:
                if result.returncode != 0:
                    failures += 1
                    print(
                        f"chunk {chunk_index} exited with {result.returncode}; continuing",
                        flush=True,
                    )
                break
            restarts += 1
            if restarts > args.max_device_restarts:
                print(
                    f"device faults exceeded restart limit while measuring chunk {chunk_index}",
                    flush=True,
                )
                return 3
            print(
                f"device fault while measuring chunk {chunk_index}; restarting fresh process "
                f"({restarts}/{args.max_device_restarts})",
                flush=True,
            )
    print(f"all requested ops visited; child_failures={failures}", flush=True)
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
