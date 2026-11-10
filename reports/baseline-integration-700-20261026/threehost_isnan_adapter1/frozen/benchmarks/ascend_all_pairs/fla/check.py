"""Use the original checker unchanged, adding isolated providers and launch evidence."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT.parent)]
spec = importlib.util.spec_from_file_location("_fla_original_checker", ROOT / "scripts/ascend_classic_check.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def main():
    checker.require_guard()
    from benchmarks.ops import bench_ascend_classics as registry
    from benchmarks.ascend_all_pairs.fla import PROVIDERS
    from benchmarks.ascend_baselines import source_hashes
    import torch
    events = []
    original = registry.candidate_factories

    def factories(case):
        name = sys.argv[sys.argv.index("--op") + 1]
        return {"flash-linear-attention-npu": lambda: PROVIDERS[name](case)} if name in PROVIDERS else original(case)

    registry.candidate_factories = factories
    # A Python profiler records actual original JIT dispatch/native ACLNN enqueue,
    # without substituting or changing any implementation.
    def trace(frame, event, arg):
        if event == "call":
            filename = frame.f_code.co_filename
            name = frame.f_code.co_name
            if ("flash-linear-attention-npu" in filename or "/fla_npu/" in filename or
                    filename.endswith("triton/runtime/jit.py")):
                if name in ("run", "__call__", "_run", "_run_op", "_execute", "_call_aclnn") or name.startswith(("chunk_local_cumsum", "npu_")):
                    record = {"file": filename, "line": frame.f_code.co_firstlineno, "function": name}
                    obj = frame.f_locals.get("self")
                    fn = getattr(obj, "fn", None)
                    if fn is not None and hasattr(fn, "__code__"):
                        record["jit_source"] = fn.__code__.co_filename
                        record["jit_function"] = fn.__name__
                    if record not in events:
                        events.append(record)
        return trace

    sys.setprofile(trace)
    try:
        rc = checker.main()
    finally:
        sys.setprofile(None)
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    row = json.loads(output.read_text())
    row["call_attribution"] = events
    row["isolated_provider_sha256"] = source_hashes(Path(__file__), Path(__file__).with_name("__init__.py"), Path(__file__).with_name("providers.py"))
    output.write_text(json.dumps(row, indent=2) + "\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
