"""Foreground execution and receipts only; no scheduling or agent management."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

here = Path(__file__).resolve().parent
case_id, timeout, *command = sys.argv[1:]
assert "/" not in case_id and case_id not in {".", ".."}
out = here / case_id
out.mkdir(exist_ok=True)
assert not (out / "guard.json").exists()
owner = "baseline-single-agent-local-20261027/" + case_id
cmd = ["npu-guard", "run", "--owner", owner, "--wait", "120", "--timeout", timeout, "--", *command]
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
begin = time.monotonic()
(out / "COMMAND.json").write_text(json.dumps({"command": cmd, "started_at": started}, indent=2))
with (out / "guard.log").open("w") as log:
    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
meta = json.loads(Path("/run/lock/ascend-npu/device-0.lock").read_text())
receipt = {"command": cmd, "started_at": started,
           "ended_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "wall_seconds": time.monotonic() - begin, "returncode": result.returncode,
           "guard_metadata": meta if meta.get("owner") == owner else None}
(out / "guard.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"case_id": case_id, "returncode": result.returncode,
                  "wall_seconds": round(receipt["wall_seconds"], 3),
                  "checker_exists": (out / "checker.json").exists()}))
if result.returncode:
    print((out / "guard.log").read_text()[-1800:])
raise SystemExit(result.returncode)
