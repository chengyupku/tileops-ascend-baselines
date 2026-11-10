"""Parent orchestrator; each single-expert query gets a separate foreground guard."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

P = Path(__file__).resolve().parent
owner = json.loads((P/'PARENT_RELEASE.json').read_text())['owner']
env_script = P.parent/'fp16_local_remaining3/environment.sh'
rows = []
for row in json.loads((P/'CASES.json').read_text()):
    out = P/row['name']
    out.mkdir(exist_ok=False)
    runtime = tempfile.mkdtemp(prefix='baseline700-moe-workspace-')
    env = dict(os.environ, BF16_GEMM_GUARD_OWNER=owner, BF16_GEMM_RUN_ROOT=runtime)
    cmd = ['npu-guard','run','--owner',owner,'--wait','600','--timeout','120','--',
           'bash',str(env_script),'/root/dyq/workspace/.venv/bin/python','-B',str(P/'probe.py'),row['name'],str(out)]
    receipt = dict(command=cmd, requested_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   runtime=runtime, environment_script_sha256=hashlib.sha256(env_script.read_bytes()).hexdigest(),
                   probe_sha256=hashlib.sha256((P/'probe.py').read_bytes()).hexdigest())
    (out/'REQUEST.json').write_text(json.dumps(receipt,indent=2)+'\n')
    with (out/'guard.stdout').open('w') as stdout, (out/'guard.stderr').open('w') as stderr:
        start=time.monotonic()
        proc=subprocess.run(cmd,env=env,stdout=stdout,stderr=stderr)
    receipt.update(exit_code=proc.returncode, elapsed_seconds=time.monotonic()-start,
                   finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    try:
        guard=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
        if guard['owner']==owner:receipt['guard_metadata']=guard
    except (OSError,ValueError,KeyError):pass
    (out/'GUARD_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    result=json.loads((out/'RESULT.json').read_text()) if (out/'RESULT.json').exists() else None
    rows.append(dict(name=row['name'],exit_code=proc.returncode,result=result,guard_receipt=str(out/'GUARD_RECEIPT.json')))
    (P/'RESULTS.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(dict(name=row['name'],exit_code=proc.returncode,workspace_bytes=result.get('workspace_bytes') if result else None)),flush=True)
    if proc.returncode!=0:break
