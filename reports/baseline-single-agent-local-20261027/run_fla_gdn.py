"""Execute one previously unrun local GDN case using the established contract."""
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
case_id = sys.argv[1]
assert case_id in {
    "q35-gva-decode-b8-float16", "q35-gva-decode-b64-float16",
    "q35-gva-decode-b256-float16", "q35-gva-decode-b512-float16",
}
meta = json.loads(Path("/run/lock/ascend-npu/device-0.lock").read_text())
assert meta["phase"] == "running" and meta["child"]["pid"] == os.getpgrp()
assert meta["owner"] == "baseline-single-agent-local-20261027/" + case_id
out = HERE / case_id
out.mkdir(exist_ok=True)
assert not (out / "checker.json").exists()
os.chdir(out)
(out / "GUARD_RUNNING.json").write_text(json.dumps(meta, indent=2))
short = Path(f"/proc/{os.getpid()}/cwd")
for name in ("tmp", "ASCEND_WORK_PATH", "ASCEND_CACHE_PATH", "ASCEND_PROCESS_LOG_PATH",
             "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
    (out / name).mkdir(exist_ok=True)
    os.environ[name if name != "tmp" else "TMPDIR"] = str(short / name)
tempfile.tempdir = os.environ["TMPDIR"]
sys.path[:0] = [str(WORKSPACE / "TileOPs"), str(WORKSPACE / "TileOPs/src"), str(WORKSPACE)]
preflight = json.loads((HERE / "FLA_GDN_PREFLIGHT.json").read_text())
for filename, expected in preflight["source_sha256"].items():
    assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
adapter = Path(preflight["adapter"])
spec = importlib.util.spec_from_file_location("single_agent_existing_gdn", adapter)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from benchmarks.ascend_all_pairs import fla


def bind(case):
    candidate = module.build(case)
    return replace(candidate, source_sha256=candidate.source_sha256 | {
        str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })


fla.PROVIDERS["GDNFwdOp"] = bind
checker = Path(preflight["checker"])
sys.argv = [str(checker), "--op", "GDNFwdOp", "--case-id", case_id,
            "--library", "flash-linear-attention-npu", "--output", str(out / "checker.json")]
try:
    runpy.run_path(str(checker), run_name="__main__")
finally:
    (out / "maps.txt").write_text(Path("/proc/self/maps").read_text())
    import torch
    if torch.npu.is_initialized():
        torch.npu.synchronize()
