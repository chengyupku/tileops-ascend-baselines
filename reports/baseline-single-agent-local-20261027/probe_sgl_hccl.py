"""Guarded local communication-dependency probe; not baseline acceptance."""
import datetime
import json
import os
from pathlib import Path
import traceback

HERE = Path(__file__).resolve().parent
OUT = HERE / "sgl-hccl-local-probe01"
meta = json.loads(Path("/run/lock/ascend-npu/device-0.lock").read_text())
assert meta["phase"] == "running" and meta["child"]["pid"] == os.getpgrp()
assert meta["owner"] == "baseline-single-agent-local-20261027/sgl-hccl-local-probe01"
(OUT / "GUARD_RUNNING.json").write_text(json.dumps(meta, indent=2))
os.chdir(OUT)
for name in ("ASCEND_PROCESS_LOG_PATH", "ASCEND_WORK_PATH", "ASCEND_CACHE_PATH"):
    path = OUT / name
    path.mkdir(exist_ok=True)
    os.environ[name] = str(path)
row = {"diagnostic_only": True, "baseline_case_accepted": False,
       "rootinfo_exists": Path("/etc/hccl_rootinfo.json").exists(), "world_size": 1}
stage = "import"
dist = None
try:
    import torch
    import torch_npu
    import torch.distributed as dist
    row["torch"] = torch.__file__
    row["torch_npu"] = torch_npu.__file__
    torch.npu.set_device(0)
    stage = "init_process_group"
    dist.init_process_group("hccl", init_method="file://" + str(OUT / "rendezvous"),
                            rank=0, world_size=1, timeout=datetime.timedelta(seconds=40))
    stage = "all_to_all_single"
    x = torch.arange(16, dtype=torch.float32, device="npu")
    out = torch.empty_like(x)
    torch.npu.synchronize()
    print("ORIGINAL_HCCL_ALLTOALL_ENTERED", flush=True)
    dist.all_to_all_single(out, x)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), x.cpu(), rtol=0, atol=0)
    row["status"] = "hccl_world1_alltoall_passed"
except Exception as exc:
    row.update(status="dependency_probe_failed", stage=stage,
               error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
finally:
    (OUT / "RESULT.json").write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps({k: row[k] for k in ("status", "stage", "error") if k in row}), flush=True)
    if dist is not None and dist.is_initialized():
        dist.destroy_process_group()
raise SystemExit(0 if row["status"] == "hccl_world1_alltoall_passed" else 1)
