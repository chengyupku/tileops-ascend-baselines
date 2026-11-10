import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
OUT = HERE / "sum-aclnn-isolated-02"
meta = json.loads(Path("/run/lock/ascend-npu/device-0.lock").read_text())
assert meta["phase"] == "running" and meta["child"]["pid"] == os.getpgrp()
assert meta["owner"] == "baseline-single-agent-local-20261027/sum-aclnn-isolated-02"
assert not (OUT / "checker.json").exists()
(OUT / "GUARD_RUNNING.json").write_text(json.dumps(meta, indent=2))
pins = json.loads((HERE / "SUM_ISOLATED_PREFLIGHT.json").read_text())["source_sha256"]
for filename, digest in pins.items():
    assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest, filename
for name in ("ASCEND_PROCESS_LOG_PATH", "ASCEND_CACHE_PATH", "ASCEND_WORK_PATH",
             "XDG_CACHE_HOME", "TMPDIR", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR"):
    path = OUT / name
    path.mkdir(exist_ok=True)
    os.environ[name] = str(path)
os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "0"
os.chdir(OUT)
sys.path[:0] = [str(HERE), str(WORKSPACE / "TileOPs"),
               str(WORKSPACE / "TileOPs/src"), str(WORKSPACE)]
from benchmarks.ascend_all_pairs.ops_math import PROVIDERS
from sum_aclnn_adapter import factory
PROVIDERS["SumFwdOp"] = lambda case: factory(case, OUT, pins)
checker = WORKSPACE / "TileOPs/scripts/ascend_pair_check.py"
sys.argv = [str(checker), "--op", "SumFwdOp", "--case-id", "mlp-intermediate-float32",
            "--library", "ops-math", "--output", str(OUT / "checker.json")]
runpy.run_path(str(checker), run_name="__main__")
