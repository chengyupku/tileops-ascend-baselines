"""One guarded, process-limited foreground experiment; never submits to remote hosts."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
label,script,*args=sys.argv[1:]
script_path=Path(script)
if not script_path.is_absolute():
    script_path=HERE/script_path
if not script_path.is_file():
    raise FileNotFoundError(script_path)
out=HERE/'requests'/label
out.mkdir(parents=True,exist_ok=False)
command=['npu-guard','run','--owner','main/elementwise-local32/'+label,'--wait','600','--timeout','900','--receipt',str(out/'guard.json'),'--','bash','-c','source "$1"\nshift\nexec "$@"','local32',str(HERE/'env.sh'),str(ROOT/'.venv/bin/python'),str(ROOT/'tools/run_with_npu_core_limit.py'),'--cube','32','--vector','64','--script',str(script_path.resolve()),'--',*args]
(out/'command.json').write_text(json.dumps(command,indent=2)+'\n')
start=time.monotonic()
with (out/'output.log').open('w') as f:
 code=subprocess.call(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
(out/'status.json').write_text(json.dumps({'guard_exit_code':code,'wall_seconds':time.monotonic()-start,'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runtime_limit':{'cube':32,'vector':64}},indent=2)+'\n')
raise SystemExit(code)
