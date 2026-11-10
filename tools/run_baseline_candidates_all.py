#!/usr/bin/env python3
"""Run candidate-baseline profile workers in NPU-guarded, resumable chunks."""

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
MEASURE = ROOT / "tools" / "measure_baseline_candidates.py"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument(
        "--chunk-size",
        type=int,
        default=8,
        help="number of ops per guarded worker; larger values amortize NPU/process setup",
    )
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--max-device-restarts", type=int, default=32)
    p.add_argument("--gc-every", type=int, default=8)
    p.add_argument("--sync-every", type=int, default=32)
    p.add_argument(
        "--torch-compile-fallback-only",
        action="store_true",
        help="measure the independent torch.compile fallback for cases missing that candidate",
    )
    p.add_argument("--start-chunk", type=int, default=0)
    p.add_argument("--end-chunk", type=int)
    p.add_argument(
        "--cases-file",
        type=Path,
        help="restrict to (op, case_id) pairs listed here; only ops with a listed case are "
        "chunked, and the file is passed through to each worker",
    )
    args = p.parse_args()
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    sys.path.insert(0, str(ROOT / "TileOPs" / "src"))
    sys.path.insert(0, str(ROOT / "TileOPs"))
    from benchmarks._cases import registered
    from tileops.manifest import load_manifest

    names = sorted(set(load_manifest()) & set(registered()))
    if args.cases_file:
        filter_ops = set()
        for raw in args.cases_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "::" not in line:
                continue
            filter_ops.add(line.split("::", 1)[0].strip())
        names = [name for name in names if name in filter_ops]
    chunks = [names[i : i + args.chunk_size] for i in range(0, len(names), args.chunk_size)]
    print(f"profiling all benchmark candidates for {len(names)} ops into {args.output}", flush=True)
    failures = 0
    for chunk_index, chunk in enumerate(chunks):
        if chunk_index < args.start_chunk:
            continue
        if args.end_chunk is not None and chunk_index >= args.end_chunk:
            break
        owner = f"root/baseline-candidates/chunk-{chunk_index:03d}"
        child = (
            f"source {shlex.quote(str(ROOT / 'env.sh'))}; "
            f"export PYTHONPATH={shlex.quote(str(ROOT / 'TileOPs' / 'src'))}:"
            f"{shlex.quote(str(ROOT / 'TileOPs'))}:$PYTHONPATH; "
            f"exec {shlex.quote(sys.executable)} {shlex.quote(str(MEASURE))} "
            + " ".join(f"--op {shlex.quote(op)}" for op in chunk)
            + f" --warmup {args.warmup} --repeats {args.repeats}"
            + f" --gc-every {args.gc_every} --sync-every {args.sync_every}"
            + (" --torch-compile-fallback-only" if args.torch_compile_fallback_only else "")
            + (f" --cases-file {shlex.quote(str(args.cases_file))}" if args.cases_file else "")
            + f" --output {shlex.quote(str(args.output))}"
        )
        print(
            f"===== candidate chunk {chunk_index + 1}/{len(chunks)}: {', '.join(chunk)} ({owner}) =====",
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
                    return 78
                time.sleep(5)
                continue
            if result.returncode != 3:
                if result.returncode:
                    failures += 1
                    print(
                        f"candidate chunk {chunk_index} exited with {result.returncode}; continuing",
                        flush=True,
                    )
                break
            restarts += 1
            if restarts > args.max_device_restarts:
                return 3
            print(
                f"candidate device fault in chunk {chunk_index}; restarting "
                f"({restarts}/{args.max_device_restarts})",
                flush=True,
            )
    print(f"all candidate ops visited; child_failures={failures}", flush=True)
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
