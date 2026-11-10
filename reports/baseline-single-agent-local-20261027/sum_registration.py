"""One verified local ReduceSum case; no runtime import during construction."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent


def _plain(value):
    from collections.abc import Mapping
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def install(providers):
    # The import shim pins this registration and both manifests before installation.
    rows = [json.loads((ROOT / "SUM_VERIFIED_CASE.json").read_text())]
    contracts = {row["key"][2]: row["snapshot"] for row in rows}
    pins = json.loads((ROOT / "SUM_ISOLATED_PREFLIGHT03.json").read_text())["source_sha256"]
    previous = providers.get("SumFwdOp")

    def make(case):
        from benchmarks.ascend_baselines import Candidate, UnsupportedCase
        if case.id not in contracts:
            if previous is not None:
                return previous(case)
            raise UnsupportedCase("No matching original ReduceSum route.")
        contract = contracts[case.id]
        if (case._call.signature.name != "SumFwdOp"
                or _plain(case.params) != contract["parameters"]
                or _plain(case._call.tensors) != contract["declared_tensors"]):
            raise UnsupportedCase("ReduceSum case differs from the fully verified contract.")
        child = None
        guard_id = None
        closed = False

        def guard():
            meta = json.loads(Path("/run/lock/ascend-npu/device-0.lock").read_text())
            if meta["phase"] != "running" or meta["child"]["pid"] != os.getpgrp():
                raise RuntimeError("Foreground npu-guard must cover the complete runtime.")
            return meta["pid"], meta["start_ticks"], meta["child"]["pid"], meta["child"]["start_ticks"]

        def cleanup():
            nonlocal child, closed
            if child is not None:
                if guard() != guard_id:
                    raise RuntimeError("Cleanup requires the original guard lifetime.")
                child.cleanup()
                child = None
            closed = True

        def run(*args, **kwargs):
            nonlocal child, guard_id
            if closed:
                raise RuntimeError("Candidate already cleaned up.")
            current = guard()
            if guard_id is not None and current != guard_id:
                raise RuntimeError("Candidate cannot cross guard lifetimes.")
            if child is None:
                vendor = str(ROOT / "sum-original-install/vendors/single_sum_math")
                if any(os.environ.get(n) != vendor for n in
                       ("ASCEND_CUSTOM_OPP_PATH", "ASCEND_CLASSIC_MATH_VENDOR")):
                    raise RuntimeError("Select " + str(ROOT / "ops-math-registered-python")
                                       + " before importing the device runtime.")
                for name, expected in pins.items():
                    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                        raise RuntimeError("Verified source or runtime changed: " + name)
                path = ROOT / "sum_graph_isolated_adapter_v2.py"
                spec = importlib.util.spec_from_file_location("_single_sum_graph", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                output = ROOT / "registered-runs"
                output.mkdir(exist_ok=True)
                output = Path(tempfile.mkdtemp(prefix=case.id + "-", dir=output))
                guard_id = current
                child = module.factory(case, output, pins)
            return child.implementation.run(*args, **kwargs)

        return Candidate(SimpleNamespace(run=run, args=None, reset=None, noncomparable_reason=None),
                         "ops-math::ReduceSum via aclopCompileAndExecute",
                         "Original complete graph ReduceSum, axis -1, FP32, unchanged layout; "
                         "isolated original package excludes unrelated custom operators.",
                         dict(pins), limitations=("Requires the verified local vendor environment.",
                                                 "No performance measurement claimed."),
                         timing_mode="host_roundtrip", cleanup=cleanup)

    make.previous = previous
    providers["SumFwdOp"] = make
