#!/usr/bin/env bash
set -euo pipefail
ssh openlibing-DevEnv-055409 'python3 - <<'"'"'PY'"'"'
from pathlib import Path
import json
b=Path("/opt/tileops-ci/server2-20261030/overnight-20261101/ci-full-v3");o=b/"measurements"
plan=json.loads((b/"workspace/ELEMENTWISE_PLAN.json").read_text());sizes={x["op"]:len(x["case_ids"]) for x in plan};passed=[];failed=[]
for e in plan:
 d=o/e["op"]
 try:
  if (d/"LATEST_ATTEMPT.json").exists():d=o/json.loads((d/"LATEST_ATTEMPT.json").read_text())["path"]
  if (d/"receipt.json").exists():
   r=json.loads((d/"receipt.json").read_text());(passed if r["status"]=="passed" else failed).append(e["op"])
 except (OSError,ValueError):pass
v=dict(passed_operators=len(passed),passed_cases=sum(sizes[x] for x in passed),failed=failed)
g=json.loads(Path("/run/lock/ascend-npu/device-0.lock").read_text());v["guard_phase"]=g["phase"];v["guard_valid_so_far"]=g.get("performance_valid")
if g["owner"].startswith("ci/elementwise32/ci-full-v3/"):
 op=g["owner"].split("/")[-2];d=o/op
 try:
  child=g.get("child",{}).get("pid")
  if g["phase"]=="running" and child:
   args=Path(f"/proc/{child}/cmdline").read_bytes().decode().split("\0")
   if "--output" in args:
    candidate=Path(args[args.index("--output")+1])
    if candidate.resolve().is_relative_to(o.resolve()):d=candidate
 except OSError:pass
 rows=[]
 if (d/"measurements.jsonl").exists():
  for line in (d/"measurements.jsonl").read_text().splitlines():
   try:rows.append(json.loads(line))
   except ValueError:pass
 v["active"]=dict(op=op,case_rows=len({r["case_id"] for r in rows}),expected=sizes.get(op),last_case=rows[-1]["case_id"] if rows else None)
v["controller"]=json.loads((o/"STATE.json").read_text())
print(json.dumps(v))

PY'
