"""Replace only Torch NPU compiler autotuner timing; preserve its search space.

Search scores use one warmup and three isolated msprof windows, each preceded by
the project 1 GiB flush. They are diagnostic only. The selected compiled API is
subsequently measured by baselines.py with the formal 3/15 protocol.
"""
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import weakref


def install(torch, torch_npu, output_dir):
    from torch_npu._inductor.runtime import triton_heuristics as heuristics
    from tools.measure_baselines import _time_reference

    if getattr(heuristics, "_arithmetic9_msprof_installed", False):
        return heuristics._arithmetic9_msprof_receipt
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"search-{os.getpid()}-{time.time_ns()}.jsonl"
    lock = threading.RLock()
    # The original compiler passes the same frozen closure to its screening and
    # selected-candidate stages. Each closure already has three msprof samples;
    # reuse that search score once, with explicit provenance, rather than profile
    # the identical candidate twice. No cache survives the closure's lifetime.
    screened = weakref.WeakKeyDictionary()
    source = Path(heuristics.__file__)
    receipt = {
        "record": "installation",
        "compiler_source": str(source),
        "compiler_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "hook_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "search_warmup": 1,
        "search_repeats": 3,
        "search_only": True,
        "screen_score_reuse": "one use for identical selected closure only",
        "log": str(output),
        "unchanged": "candidate generation, compilation, selection, argument reset, kernel math and launch configuration",
    }
    output.write_text(json.dumps(receipt) + "\n")

    def describe(fn):
        result = {"callable": getattr(fn, "__qualname__", type(fn).__name__)}
        defaults = getattr(fn, "__defaults__", ()) or ()
        if defaults:
            launcher = defaults[0]
            result["config"] = str(getattr(launcher, "config", "unknown"))
        if len(defaults) > 1 and isinstance(defaults[1], (tuple, list)):
            result["arguments"] = [
                {"shape": list(arg.shape), "dtype": str(arg.dtype), "stride": list(arg.stride())}
                if isinstance(arg, torch.Tensor) else str(arg)
                for arg in defaults[1]
            ]
        return result

    def measure(fn, phase):
        with lock:
            started = time.monotonic()
            row = {"record": "search_score", "phase": phase, "search_only": True,
                   "warmup": 1, "repeats": 3, **describe(fn)}
            try:
                result = _time_reference(torch, torch_npu, fn, (), 1, 3)
                row.update(status="measured", **result)
                return result["latency_ms"]
            except Exception as exc:
                row.update(status="failed", error=repr(exc))
                raise
            finally:
                row["wall_seconds"] = time.monotonic() - started
                with output.open("a") as stream:
                    stream.write(json.dumps(row, default=str) + "\n")

    def benchmark_gpu(fn, *args, **kwargs):
        return measure(fn, "single_candidate")

    def benchmark_batch(self, kernel_funcs, benchmark_label):
        if not kernel_funcs:
            return None
        scores = []
        for fn in kernel_funcs:
            with lock:
                cached = screened.pop(fn, None)
                if cached is not None:
                    score, measured_at = cached
                    with output.open("a") as stream:
                        stream.write(json.dumps({"record": "search_score_reuse",
                            "phase": "batch_" + benchmark_label, "search_only": True,
                            "reason": "identical frozen closure already measured in prerun",
                            "prerun_monotonic": measured_at, "latency_ms": score,
                            **describe(fn)}, default=str) + "\n")
                else:
                    score = measure(fn, "batch_" + benchmark_label)
            scores.append(score)
        return tuple(scores)

    def prerun(fn):
        score = measure(fn, "prerun")
        with lock:
            screened[fn] = (score, time.monotonic())
        return score

    def profiling_benchmark(fn, *args, **kwargs):
        if kwargs.get("grad_to_none"):
            raise RuntimeError("gradient-reset autotuning is outside this arithmetic task")
        score = measure(fn, "profiling_candidate")
        quantiles = kwargs.get("quantiles")
        if quantiles is not None:
            # No arithmetic caller requests quantiles; reject rather than invent them.
            raise RuntimeError("quantile benchmark not supported by this task hook")
        return score

    heuristics._measure_prerun_ms = prerun
    heuristics.benchmarker.benchmark_gpu = benchmark_gpu
    heuristics.NPUCachingAutotuner._benchmark_kernel_funcs_batch = benchmark_batch
    heuristics.do_bench_using_profiling_npu = profiling_benchmark
    heuristics._arithmetic9_msprof_installed = True
    heuristics._arithmetic9_msprof_receipt = receipt
    return receipt
