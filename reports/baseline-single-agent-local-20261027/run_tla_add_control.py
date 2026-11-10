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
OUT = HERE / "tla-original-add-control01"
OWNER = "baseline-single-agent-local-20261027/tla-original-add-control01"
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
identity = json.loads((HERE / "TLA_ADD_CONTROL_PREFLIGHT.json").read_text())["source_sha256"]
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

# Diagnostic only: the upstream example's own default dimensions and comparison.
module = native.load("examples/elementwise/elementwise_add.py", "vec_add")
print("ORIGINAL_ADD_COMPILE_ENTERED", flush=True)
kernel = tilelang.compile(module.vec_add.__wrapped__(1024, 1024, 128, 256),
                          out_idx=[-1], target="pto")
torch.manual_seed(0)
a = torch.randn(1024, 1024).npu()
b = torch.randn(1024, 1024).npu()
torch.npu.synchronize()
(OUT / "PRE_SYNC_COMPLETED.json").write_text('{"synchronized": true}\n')
actual = Path(kernel.adapter.lib._name).resolve()
(OUT / "actual_kernel.cu").write_text(kernel.get_kernel_source())
(OUT / "ACTUAL_LAUNCH.json").write_text(json.dumps({"library": str(actual),
    "sha256": hashlib.sha256(actual.read_bytes()).hexdigest(), "target": str(kernel.target)}))
c = kernel(a, b)
(OUT / "HOST_RETURNED.json").write_text('{"returned": true}\n')
torch.npu.synchronize()
(OUT / "DEVICE_SYNCHRONIZED.json").write_text('{"synchronized": true}\n')
ref_c = a + b
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
torch.npu.synchronize()
(OUT / "RESULT.json").write_text(json.dumps({"status": "upstream_add_control_passed",
    "diagnostic_only": True, "baseline_case_accepted": False, "shape": [1024,1024],
    "dtype": "float32", "target": "pto", "source": "examples/elementwise/elementwise_add.py"}, indent=2))
print("ORIGINAL_ADD_CONTROL_PASSED", flush=True)
