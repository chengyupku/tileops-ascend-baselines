"""Run the unchanged canonical checker with a private candidate binding."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
BUILD = WORKSPACE / ".cache/all-pairs/full_rollout_20261024/repairs/resume_20261025/boundary_1557/tla_pristine_build"
VENDOR = BUILD / "source"
OUT = HERE / "rmsnorm-presync01"
OWNER = "baseline-single-agent-local-20261027/rmsnorm-presync01"
meta = json.loads(Path("/run/lock/ascend-npu/device-0.lock").read_text())
assert meta["phase"] == "running" and meta["child"]["pid"] == os.getpgrp()
assert meta["owner"] == OWNER
OUT.mkdir(exist_ok=True)
assert not (OUT / "checker.json").exists()
(OUT / "GUARD_RUNNING.json").write_text(json.dumps(meta, indent=2))
os.chdir(OUT)
os.environ["INTEGRATION700_TLA_ROUTE"] = "legacy"
os.environ["TL_ROOT"] = str(VENDOR)
for name in ("TILELANG_CACHE_DIR", "TMPDIR", "XDG_CACHE_HOME", "TLA_CYTHON_CACHE_DIR",
             "ASCEND_WORK_PATH", "ASCEND_CACHE_PATH", "ASCEND_PROCESS_LOG_PATH",
             "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
    path = OUT / name
    path.mkdir(exist_ok=True)
    os.environ[name] = str(path)
tempfile.tempdir = os.environ["TMPDIR"]
sys.path[:0] = [str(HERE), str(WORKSPACE / "TileOPs"),
               str(WORKSPACE / "TileOPs/src"), str(WORKSPACE), str(VENDOR)]
identity = json.loads((HERE / "RMSNORM_PRESYNC_PREFLIGHT.json").read_text())["source_sha256"]
for name, expected in identity.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
import setuptools
from benchmarks.ascend_all_pairs.tla import native, deployment
deployment.install_cache_hook(VENDOR, Path(os.environ["TLA_CYTHON_CACHE_DIR"]))
import torch
import tilelang
assert Path(tilelang.__file__).resolve() == VENDOR / "tilelang/__init__.py"
native.VENDOR = VENDOR
native.CACHE = OUT / "jit"
native.runtime = lambda: tilelang
(OUT / "import_maps.txt").write_text(Path("/proc/self/maps").read_text())
from benchmarks.ascend_all_pairs import tla
from rmsnorm_presync_adapter import factory
tla.PROVIDERS["RMSNormFwdOp"] = lambda case: factory(case, native, tilelang, OUT, identity)
checker = WORKSPACE / "TileOPs/scripts/ascend_pair_check.py"
sys.argv = [str(checker), "--op", "RMSNormFwdOp", "--case-id", "qk-norm-head-bfloat16",
            "--library", "tilelang-ascend", "--output", str(OUT / "checker.json")]
try:
    runpy.run_path(str(checker), run_name="__main__")
finally:
    (OUT / "maps.txt").write_text(Path("/proc/self/maps").read_text())
    if torch.npu.is_initialized():
        torch.npu.synchronize()
