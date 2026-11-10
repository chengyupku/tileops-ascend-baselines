"""Private provider plus the unchanged canonical full checker."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
WORKSPACE = Path("/opt/tileops-all-pairs/workspace")
attempt, case_id = sys.argv[1:]
assert Path(attempt).name == attempt
assert os.environ.get("FP8_APPROVED_DEVICE_WORKER") == "1", "Root review and device authorization required"
OUT = HERE / attempt
assert OUT.is_dir() and not (OUT / "checker.json").exists()
meta = json.loads(Path("/run/lock/ascend-npu/device-0.lock").read_text())
assert meta["phase"] == "running" and meta["child"]["pid"] == os.getpgrp()
assert meta["owner"] == "fp8-padding-055409/" + attempt
(OUT / "GUARD_RUNNING.json").write_text(json.dumps(meta, indent=2))
pins = json.loads((HERE / "FP8_RUNTIME_PREFLIGHT.json").read_text())["source_sha256"]
assert pins, "Runtime source/library evidence must be populated before use"
for name, digest in pins.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
scope = json.loads((HERE / "SCOPE.json").read_text())
if isinstance(scope, dict):
    scope = scope.get("items", scope.get("rows", scope.get("cases", [])))
keys = [r["key"] if isinstance(r, dict) else r for r in scope]
assert ["ops-transformer", "GemmFP8FwdOp", case_id] in keys
os.chdir(OUT)
for name in ("ASCEND_PROCESS_LOG_PATH", "ASCEND_CACHE_PATH", "ASCEND_WORK_PATH",
             "XDG_CACHE_HOME", "TMPDIR", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR"):
    path = OUT / name
    path.mkdir(exist_ok=True)
    os.environ[name] = str(path)
os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "0"
sys.path[:0] = [str(HERE), str(WORKSPACE / "TileOPs"),
               str(WORKSPACE / "TileOPs/src"), str(WORKSPACE)]
from benchmarks.ascend_all_pairs.ops_transformer import PROVIDERS
from fp8_padded_adapter import factory
PROVIDERS["GemmFP8FwdOp"] = lambda case: factory(case, OUT, pins)
checker = WORKSPACE / "TileOPs/scripts/ascend_pair_check.py"
sys.argv = [str(checker), "--op", "GemmFP8FwdOp", "--case-id", case_id,
            "--library", "ops-transformer", "--output", str(OUT / "checker.json")]
runpy.run_path(str(checker), run_name="__main__")
