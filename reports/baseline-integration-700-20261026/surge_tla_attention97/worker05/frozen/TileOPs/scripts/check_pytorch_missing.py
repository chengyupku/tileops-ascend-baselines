#!/usr/bin/env python3
"""Prepare CPU oracles, then validate device PyTorch candidates without retiming CPU.

Example (from TileOPs, PYTHONPATH=src:.):
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 python scripts/check_pytorch_missing.py prepare --phase first --root PATH
  npu-guard run --owner 'main/pytorch11' --wait 600 --timeout 600 -- \
    python scripts/check_pytorch_missing.py execute --phase first --device npu --root PATH

Use --phase all only after first succeeds. CPU work never initializes the NPU.
Every row records actual full-output validation; elapsed time is diagnostic,
not a benchmark score. This does not modify the concurrently running baseline
inventory or publish a performance result.
"""

import argparse
import gc
import hashlib
import importlib.abc
import importlib.util
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "execute"))
    parser.add_argument("--phase", choices=("first", "all"), default="first")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "npu"), default="cpu")
    parser.add_argument("--op", action="append")
    parser.add_argument("--case")
    parser.add_argument("--compile-backend")
    parser.add_argument("--skip-cpu-candidate", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.mode == "prepare" or args.device == "cpu":
        os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        # Dynamo discovers installed device plugins even for an eager CPU graph.
        # Hide accelerator backends before importing torch/Dynamo; CPU phases
        # must not initialize a runtime outside the host NPU reservation.
        blocked = {"torch_npu", "acl", "aclnn", "torchair", "fla", "ascend"}

        class DenyAccelerator(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in blocked:
                    raise ImportError("CPU phase forbids device runtime: " + fullname)

        sys.meta_path.insert(0, DenyAccelerator())
        find_spec = importlib.util.find_spec

        def optional_spec(name, *a, **kw):
            return None if name.split(".")[0] in blocked else find_spec(name, *a, **kw)

        importlib.util.find_spec = optional_spec
    import torch

    from benchmarks.pytorch_missing import FUNCTIONS, bind
    from benchmarks.pytorch_missing.validation import (
        MUTATED,
        calls,
        evidence,
        inputs_for,
        reference,
        workload,
    )
    from workloads.numerics import compare_outputs

    torch.set_num_threads(args.threads)
    if args.mode == "execute" and args.device == "npu":
        import torch_npu  # noqa: F401 -- register NPU; process must be inside npu-guard.

        torch.npu.set_device(0)
        torch.npu.config.allow_internal_format = False
        torch.npu.matmul.allow_hf32 = False
    args.root.mkdir(parents=True, exist_ok=True)
    manifest = []
    failures = 0
    sources = list(
        (Path(__file__).resolve().parents[1] / "benchmarks/pytorch_missing").glob("*.py")
    )
    sources += [Path(__file__).resolve()]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for op in args.op or FUNCTIONS:
        selected = calls(op)
        if args.case:
            selected = [c for c in selected if c.case_id == args.case]
        elif args.phase == "first":
            selected = selected[:1]
        for call in selected:
            fixture = args.root / "fixtures" / op / f"{call.case_id}.pt"
            row = dict(
                op=op,
                case=call.case_id,
                phase=args.phase,
                mode=args.mode,
                device=args.device,
                compile_backend=args.compile_backend,
                source_sha256=hashes,
            )
            started = time.monotonic()
            print(
                json.dumps(dict(event="start", op=op, case=call.case_id, mode=args.mode)),
                flush=True,
            )
            try:
                w = workload(op, call)
                if args.mode == "prepare":
                    torch.manual_seed(20261022)
                    inputs = inputs_for(op, w)
                    expected = reference(op, w, inputs)
                    payload = dict(
                        op=op,
                        case=call.case_id,
                        params=call.params,
                        inputs=inputs,
                        expected=expected,
                    )
                    fixture.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(payload, fixture)
                    if args.skip_cpu_candidate:
                        row.update(status="prepared", full_output_check=False)
                    else:
                        actual_inputs = tuple(
                            t.clone() if i == MUTATED.get(op) else t for i, t in enumerate(inputs)
                        )
                        versions = [
                            t._version if isinstance(t, torch.Tensor) else None
                            for t in actual_inputs
                        ]
                        with torch.no_grad():
                            got = bind(op, call.params)(*actual_inputs)
                        check = compare_outputs(got, expected, evidence(op, w, inputs))
                        for i, t in enumerate(actual_inputs):
                            if isinstance(t, torch.Tensor) and i != MUTATED.get(op):
                                assert t._version == versions[i], (
                                    f"Unexpected mutation of input {i}"
                                )
                        row.update(status="passed", full_output_check=True, check=asdict(check))
                else:
                    payload = torch.load(fixture, map_location="cpu", weights_only=True, mmap=True)
                    assert payload["params"] == call.params
                    inputs, expected = payload["inputs"], payload["expected"]
                    actual_inputs = tuple(
                        t.to(args.device).clone() if isinstance(t, torch.Tensor) else t
                        for t in inputs
                    )
                    versions = [
                        t._version if isinstance(t, torch.Tensor) else None for t in actual_inputs
                    ]
                    fn = bind(op, call.params)
                    if args.compile_backend:
                        # Each case is an independent compilation experiment;
                        # partial-call wrappers otherwise share Dynamo's finite
                        # recompile budget across unrelated operator functions.
                        torch._dynamo.reset()
                        # Manifest cases have fixed shapes. Dynamic shape inference
                        # can be pathological for repeated butterfly reshapes.
                        fn = torch.compile(
                            fn, backend=args.compile_backend, fullgraph=True, dynamic=False
                        )
                    if args.device == "npu":
                        torch.npu.synchronize()
                    run_start = time.monotonic()
                    with torch.no_grad():
                        got = fn(*actual_inputs)
                    if args.device == "npu":
                        torch.npu.synchronize()
                    row["execution_seconds"] = time.monotonic() - run_start
                    got = (
                        tuple(t.cpu() for t in got) if isinstance(got, (tuple, list)) else got.cpu()
                    )
                    check = compare_outputs(got, expected, evidence(op, w, inputs))
                    for i, t in enumerate(actual_inputs):
                        if isinstance(t, torch.Tensor) and i != MUTATED.get(op):
                            assert t._version == versions[i], f"Unexpected mutation of input {i}"
                    row.update(status="passed", full_output_check=True, check=asdict(check))
                # Drop multi-GB fixtures between cases, including mmap references.
                actual_inputs = inputs = expected = payload = got = t = None
                gc.collect()
                if args.mode == "execute" and args.device == "npu":
                    torch.npu.synchronize()
                    torch.npu.empty_cache()
            except Exception as error:
                failures += 1
                row.update(
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(),
                )
            row["wall_seconds"] = time.monotonic() - started
            manifest.append(row)
            name = f"{args.mode}_{args.phase}_{args.device}_{args.compile_backend or 'eager'}.jsonl"
            with (args.root / name).open("a") as log:
                log.write(json.dumps(row) + "\n")
            print(
                json.dumps(
                    {k: v for k, v in row.items() if k not in ("source_sha256", "traceback")}
                ),
                flush=True,
            )
    print(json.dumps(dict(total=len(manifest), failed=failures)), flush=True)
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
