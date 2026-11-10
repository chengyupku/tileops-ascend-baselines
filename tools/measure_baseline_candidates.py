#!/usr/bin/env python3
"""Profile every baseline implementation exposed by the TileOPs benchmark cases.

The ordinary benchmark runner constructs a mapping of named implementations inside each
``test_*_bench`` function.  This command reuses those factories, replaces the CUDA timing
runner with a capture runner, and measures the captured callables with the same
``torch_npu.profiler.profile`` protocol as ``measure_baselines.py``.  A separate JSONL is
used because a workload can have several candidate rows.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import gc
import glob
import importlib
import json
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TILEOPS_ROOT = ROOT / "TileOPs"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TILEOPS_ROOT / "src"))
sys.path.insert(0, str(TILEOPS_ROOT))

from tools.baseline_protocol import TERMINAL_FAILURES, begin, current_row, provenance, seed_case

from tools.measure_baselines import (  # noqa: E402
    _jsonable,
    _environment,
    _git_head,
    _tensor_digest,
    _time_reference,
)


class _JsonlAppender:
    """Keep one append handle per worker and sync periodically.

    The old path reopened and ``fsync``-ed the JSONL file for every candidate.  That
    is useful for a tiny one-shot experiment, but becomes visible host overhead when
    the candidate sweep emits tens of thousands of rows.  A worker is already the
    sole writer for its chunk, so line-buffered writes plus a periodic sync preserve
    resumability while avoiding one open/fsync round-trip per candidate.  The final
    close always flushes and syncs the remaining rows.
    """

    def __init__(self, path: Path, sync_every: int = 32):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")
        self._sync_every = max(1, sync_every)
        self._writes = 0

    def append(self, row: dict[str, Any]) -> None:
        provenance(row)
        self._file.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")
        self._file.flush()
        self._writes += 1
        if self._writes % self._sync_every == 0:
            os.fsync(self._file.fileno())

    def close(self) -> None:
        if self._file is None:
            return
        try:
            self._file.flush()
            os.fsync(self._file.fileno())
        finally:
            self._file.close()
            self._file = None


def _discover_factories() -> tuple[dict[tuple[str, str], list[tuple[Any, str]]], list[dict]]:
    """Index pytest benchmark functions by manifest op and case id."""
    from benchmarks import api

    factories: dict[tuple[str, str], list[tuple[Any, str]]] = defaultdict(list)
    import_errors: list[dict] = []
    for path in sorted((TILEOPS_ROOT / "benchmarks" / "ops").glob("bench_*.py")):
        module_name = f"benchmarks.ops.{path.stem}"
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            import_errors.append(
                {
                    "module": module_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        for function_name, function in vars(module).items():
            if not function_name.startswith("test_") or not callable(function):
                continue
            marks = getattr(function, "pytestmark", [])
            if not isinstance(marks, list):
                marks = [marks]
            for mark in marks:
                mark_args = getattr(mark, "args", ())
                if getattr(mark, "name", None) != "parametrize" or len(mark_args) < 2:
                    continue
                parameter_names, marked_cases = mark_args[:2]
                if parameter_names != "case":
                    continue
                for marked_case in marked_cases:
                    try:
                        op = marked_case._call.signature.name
                    except Exception:
                        continue
                    factories[(op, marked_case.id)].append((module, function_name))
    return factories, import_errors


class _CaptureRunner:
    captures: list[tuple[Any, Any]] = []

    def __init__(self, _op: Any, case: Any):
        self.case = case

    def compare(self, implementations: Any) -> None:
        self.captures.append((self.case, implementations))


def _implementation_items(value: Any):
    from benchmarks.api import Implementation

    if isinstance(value, dict):
        return value.items()
    if isinstance(value, Implementation):
        return [("baseline", value)]
    return [("baseline", value)]


def _impl_parts(value: Any, case: Any):
    from benchmarks.api import Implementation

    if isinstance(value, Implementation):
        fn = value.run
        args = case.inputs if value.args is None else tuple(value.args)
        reset = value.reset
    else:
        fn = value
        args = case.inputs
        reset = None
    return fn, tuple(args), reset


def _load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.open(encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or not row.get("op") or not row.get("case_id"):
            continue
        candidate = row.get("candidate")
        if not current_row(row):
            continue
        if row.get("status") == "invalidated_external_activity":
            done.discard(f"{row.get('op')}::{row.get('case_id')}::{row.get('candidate')}")
            continue
        if row.get("status") in TERMINAL_FAILURES:
            done.add(f"{row['op']}::{row['case_id']}::{row.get('candidate')}")
            continue
        if row.get("status") in {
            "measured_candidate",
            "measured_candidate_input_mutated",
            "device_fault",
            "no_implementation",
            # Finding 3: a correctness failure is a terminal outcome for this candidate tag
            # too, not a transient error -- otherwise every resume re-verifies (and re-fails)
            # the same known-wrong candidate forever. --force-candidate still re-attempts it,
            # same as any other terminal status.
            "failed_verification",
        } and candidate:
            done.add(f"{row['op']}::{row['case_id']}::{candidate}")
    return done


def _profile_candidate(torch, torch_npu, fn, args, reset, warmup, repeats):
    def invoke(*_unused):
        return fn(*args)

    # reset is passed through rather than folded into invoke() here, so _time_reference can
    # call it, flush, *then* fn() and exclude reset's own device time by name (Finding 2) --
    # folding it in as before gave _parse_ffts_profile no way to tell the two apart.
    return _time_reference(torch, torch_npu, invoke, (), warmup, repeats, reset=reset)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--op", action="append")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--case-regex")
    parser.add_argument("--fallback-allow-file", type=Path,
                        help="freeze automatic compile fallback to listed op::case pairs")
    parser.add_argument(
        "--cases-file",
        type=Path,
        help="only process (op, case_id) pairs listed here, one 'op::case_id' per line "
        "(blank lines and '#' comments ignored); combines with --op/--case-regex",
    )
    parser.add_argument(
        "--torch-compile-fallback-only",
        action="store_true",
        help="skip benchmark candidate factories and measure only the independent torch.compile fallback",
    )
    parser.add_argument(
        "--force-candidate",
        action="append",
        default=[],
        help="reprofile an existing candidate tag (repeatable); useful for replacing a smoke run",
    )
    parser.add_argument(
        "--sync-every",
        type=int,
        default=32,
        help="fsync the candidate JSONL after this many rows",
    )
    parser.add_argument(
        "--gc-every",
        type=int,
        default=8,
        help="run Python GC after this many candidates (0 disables periodic GC)",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be non-negative and --repeats must be positive")
    fallback_allow = None
    if args.fallback_allow_file:
        fallback_allow = set(args.fallback_allow_file.read_text().splitlines())

    cases_filter: set[tuple[str, str]] | None = None
    if args.cases_file:
        cases_filter = set()
        for raw in args.cases_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "::" not in line:
                parser.error(f"--cases-file line is not 'op::case_id': {line!r}")
            op_part, case_part = line.split("::", 1)
            cases_filter.add((op_part.strip(), case_part.strip()))
        if not cases_filter:
            parser.error("--cases-file contained no 'op::case_id' entries")

    import torch
    import torch_npu  # noqa: F401

    from benchmarks import api
    from benchmarks.api import cases
    from benchmarks._cases import registered
    from benchmarks.baselines import compiled_reference
    from tileops.manifest import load_manifest
    from workloads.device import set_run_device
    from workloads.numerics import Request, verify

    if not torch.npu.is_available():
        raise RuntimeError("torch.npu is unavailable")
    set_run_device("npu")
    manifest = load_manifest()
    if args.torch_compile_fallback_only:
        factories, import_errors = {}, []
    else:
        factories, import_errors = _discover_factories()
    _CaptureRunner.captures = []
    api.Runner = _CaptureRunner
    done = _load_done(args.output)
    if args.force_candidate:
        forced = set(args.force_candidate)
        done = {key for key in done if key.rsplit("::", 1)[-1] not in forced}
    selected = sorted(set(manifest) & set(registered()))
    if args.op:
        selected = [name for name in selected if name in set(args.op)]
    if cases_filter is not None:
        filter_ops = {op for op, _ in cases_filter}
        selected = [name for name in selected if name in filter_ops]
    env = _environment(torch, torch_npu)
    env["candidate_factory_import_errors"] = import_errors
    writer = _JsonlAppender(args.output, args.sync_every)
    atexit.register(writer.close)
    failures = 0
    profiled = 0
    verification_failures = 0
    candidates_since_gc = 0
    unavailable_factories: set[tuple[str, str]] = set()
    for op_name in selected:
        try:
            op_cases = cases(op_name)
        except Exception as exc:
            print(f"[case-build-failed] {op_name}: {exc}", file=sys.stderr, flush=True)
            continue
        for case in op_cases:
            if args.case_regex and not re.search(args.case_regex, case.id):
                continue
            if cases_filter is not None and (op_name, case.id) not in cases_filter:
                continue
            if f"{op_name}::{case.id}::__discovery__" in done:
                continue
            discovery_row = {"op": op_name, "case_id": case.id, "candidate": "__discovery__",
                             "status": "starting", "warmup": args.warmup, "repeats": args.repeats}
            begin(discovery_row, args.output, "discovery")
            input_seed = seed_case(torch, op_name, case.id)
            factory_errors = []
            funcs = factories.get((op_name, case.id), [])
            candidates: list[tuple[str, Any, str]] = []
            for module, function_name in funcs:
                factory_key = (module.__name__, function_name)
                if factory_key in unavailable_factories:
                    factory_errors.append({"factory": ".".join(factory_key), "error": "dependency unavailable in this worker"})
                    continue
                _CaptureRunner.captures = []
                try:
                    getattr(module, function_name)(case)
                except Exception as exc:
                    failures += 1
                    message = str(exc)
                    factory_errors.append({"factory": ".".join(factory_key), "error_type": type(exc).__name__, "error": message})
                    # A missing optional baseline package makes this factory fail for
                    # every case.  Cache that fact within the worker so hundreds of
                    # cases do not repeat the same import/setup exception.  Shape- or
                    # dtype-specific runtime errors are intentionally not cached.
                    dependency_failure = isinstance(exc, ModuleNotFoundError) or (
                        "install it" in message
                        or "selected baseline" in message
                        or "No module named" in message
                    )
                    if dependency_failure:
                        unavailable_factories.add(factory_key)
                    print(
                        f"[candidate-factory-failed] {op_name}/{case.id} "
                        f"{module.__name__}.{function_name}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                for _captured_case, implementations in _CaptureRunner.captures:
                    for candidate, implementation in _implementation_items(implementations):
                        candidates.append(
                            (str(candidate), implementation, f"{module.__name__}.{function_name}")
                        )
            # A benchmark factory can fail while constructing an optional baseline before it
            # reaches Runner.compare (for example, flag_gems/quack/flashinfer/vllm is absent).
            # It is incorrect to let that setup failure hide the independent PyTorch compiled
            # reference.  Some benchmark functions also intentionally expose only a vendor
            # baseline.  Every manifest case gets one explicit torch.compile attempt unless
            # the factory already supplied that tag.
            fallback_permitted = fallback_allow is None or f"{op_name}::{case.id}" in fallback_allow
            if fallback_permitted and not any(candidate == "torch-compile" for candidate, _, _ in candidates):
                try:
                    fallback_impl = compiled_reference(case.reference)
                    if op_name == "SSDRecurrentFwdOp":
                        from benchmarks.baselines import private_inputs
                        fallback_impl = private_inputs(fallback_impl, case.inputs, 5)
                    candidates.append(
                        (
                            "torch-compile",
                            fallback_impl,
                            "fallback:manifest_reference",
                        )
                    )
                    print(
                        f"[candidate-fallback] {op_name}/{case.id}/torch-compile "
                        "benchmark did not expose a usable torch-compile capture",
                        file=sys.stderr,
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"[candidate-fallback-failed] {op_name}/{case.id}: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
            # Preserve the first implementation for a tag.  Different test functions
            # occasionally repeat the same torch tag for the same workload.
            unique = {}
            for candidate, implementation, source in candidates:
                unique.setdefault(candidate, (implementation, source))
            writer.append({**discovery_row, "status": "candidate_inventory", "candidates": list(unique),
                           "factory_errors": factory_errors, "automatic_compile_fallback_permitted": fallback_permitted})
            entry = manifest[op_name]
            for candidate, (implementation, source) in unique.items():
                key = f"{op_name}::{case.id}::{candidate}"
                if key in done:
                    continue
                row = {
                    "schema": "tileops-ascend950-baseline-candidate/v1",
                    "status": "starting",
                    "op": op_name,
                    "case_id": case.id,
                    "family": entry.get("family"),
                    "manifest_status": entry.get("status"),
                    "ref_api": entry.get("ref_api"),
                    "candidate": candidate,
                    "candidate_source": source,
                    "params": case.params,
                    "environment": env,
                    "warmup": args.warmup,
                    "repeats": args.repeats,
                    "input_seed": input_seed,
                }
                begin(row, args.output)
                started = time.time()
                verification_failed = False
                try:
                    fn, impl_args, reset = _impl_parts(implementation, case)
                    row["input_digest_before"] = _tensor_digest(impl_args)
                    # Finding 3: verify before timing, the same way Runner._verify() does on
                    # the CUDA side and measure_tileops_ascend.py already does for the
                    # TileLang kernel -- this tool was the one place in the pipeline that
                    # skipped it (api.Runner was monkey-patched to _CaptureRunner, which never
                    # calls anything resembling _verify()). One candidate per verify() call,
                    # not all of a case's candidates batched into one requests={} dict: verify()
                    # raises on the first failing request and abandons the rest of the dict, so
                    # batching would let one bad candidate hide whether the others even checked.
                    try:
                        noncomparable = getattr(implementation, "noncomparable_reason", None)
                        if noncomparable:
                            row["noncomparable_reason"] = noncomparable
                            raise ValueError(f"noncomparable implementation: {noncomparable}")
                        checks = verify(
                            case.reference,
                            case.inputs,
                            evidence=case.verification,
                            requests={candidate: Request(fn, impl_args, reset, preserve_inputs=True)},
                        )
                        if not checks[candidate].checked_outputs:
                            raise ValueError(f"verification checked zero outputs: {checks[candidate].unchecked_reason}")
                    except (AssertionError, ValueError) as exc:
                        # A failed candidate must not be eligible for best_baseline (summarize_
                        # baseline_profile.py's GOOD set excludes "failed_verification"), but a
                        # wrong candidate is not a device/tooling fault: record why and let the
                        # sweep continue with the rest of this case's candidates.
                        verification_failed = True
                        row.update(
                            {
                                "status": "failed_verification",
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                    else:
                        result = checks[candidate]
                        row.update(
                            {
                                "checked_outputs": result.checked_outputs,
                                "total_outputs": result.total_outputs,
                                "max_abs_err": result.max_abs_err,
                                "unchecked_reason": result.unchecked_reason,
                            }
                        )
                    if not verification_failed:
                        profile = _profile_candidate(
                            torch, torch_npu, fn, impl_args, reset, args.warmup, args.repeats
                        )
                        row.update(
                            {
                                "status": "measured_candidate",
                                "profiler": "torch_npu.profiler.profile",
                                **profile,
                                "input_digest_after": _tensor_digest(impl_args),
                            }
                        )
                        if row["input_digest_before"] != row["input_digest_after"]:
                            row["input_mutated"] = True
                            row["status"] = "measured_candidate_input_mutated"
                        profiled += 1
                except Exception as exc:
                    failures += 1
                    message = str(exc)
                    is_device_fault = "device error type" in message or "507035" in message
                    row.update(
                        {
                            "status": "device_fault" if is_device_fault else "failed",
                            "error_type": type(exc).__name__,
                            "error": message,
                            "traceback": traceback.format_exc(limit=16),
                        }
                    )
                    if is_device_fault:
                        print("stopping after device_fault; restart in a fresh process", file=sys.stderr)
                        return 3
                finally:
                    if verification_failed:
                        verification_failures += 1
                    row["elapsed_wall_s"] = time.time() - started
                    writer.append(row)
                    candidates_since_gc += 1
                    if args.gc_every and candidates_since_gc >= args.gc_every:
                        gc.collect()
                        candidates_since_gc = 0
                    print(
                        f"{op_name}/{case.id}/{candidate} {row['status']}"
                        + (f" latency_ms={row['latency_ms']:.6f}" if "latency_ms" in row else ""),
                        flush=True,
                    )
            case._release()
            # Release allocator blocks once per workload rather than once per
            # implementation.  This keeps memory bounded while avoiding a costly
            # empty-cache/GC round-trip between every candidate of the same case.
            with contextlib.suppress(Exception):
                torch.npu.empty_cache()
    print(
        f"candidate profiling complete profiled={profiled} failures={failures} "
        f"verification_failures={verification_failures}",
        flush=True,
    )
    return 2 if (failures or verification_failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
