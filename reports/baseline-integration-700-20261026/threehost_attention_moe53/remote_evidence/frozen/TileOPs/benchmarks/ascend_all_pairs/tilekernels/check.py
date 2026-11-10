#!/usr/bin/env python3
"""Reuse unmodified full manifest checker and record original vendor call attribution."""
from pathlib import Path
import sys, os, json, hashlib
ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT.parent
CACHE = WORKSPACE / ".cache/all-pairs/tilekernels"
os.environ["TILELANG_CACHE_DIR"] = str(CACHE / "tilelang")
os.environ["TMPDIR"] = str(CACHE / "tmp")
os.environ["TRITON_CACHE_DIR"] = str(CACHE / "triton")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(WORKSPACE)]
from scripts import ascend_classic_check as checker
checker.require_guard()
from benchmarks.ascend_all_pairs.tilekernels import PROVIDERS
from benchmarks.ops import bench_ascend_classics
bench_ascend_classics.candidate_factories = lambda case: {"TileKernels": lambda: PROVIDERS[sys.argv[sys.argv.index("--op") + 1]](case)}
called = {}
vendor = str(WORKSPACE / "ascend_baselines/TileKernels/tile_kernels")
def trace(frame, event, arg):
    if event == "call" and frame.f_code.co_filename.startswith(vendor):
        path = str(Path(frame.f_code.co_filename).relative_to(WORKSPACE))
        key = (path, frame.f_code.co_firstlineno, frame.f_code.co_name)
        called[key] = called.get(key, 0) + 1
sys.setprofile(trace)
try:
    rc = checker.main()
finally:
    sys.setprofile(None)
    path = Path(sys.argv[sys.argv.index("--output") + 1])
    if path.exists():
        row = json.loads(path.read_text())
        row["provider_module"] = "benchmarks.ascend_all_pairs.tilekernels"
        row["provider_sha256"] = {str(p.relative_to(WORKSPACE)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in Path(__file__).parent.glob("*.py")}
        row["vendor_python_calls"] = [{"file": p, "line": line, "function": name, "calls": n}
                                      for (p, line, name), n in sorted(called.items())]
        row["input_preservation_requested"] = True
        row["compiler_cache"] = str(CACHE / "tilelang")
        path.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n")
sys.exit(rc)
