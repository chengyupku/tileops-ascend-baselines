"""Compatibility for installed host guard without --receipt; fail closed on races."""
import subprocess,sys,json
from pathlib import Path
args=sys.argv[1:];i=args.index('--receipt');receipt=Path(args[i+1]);del args[i:i+2]
owner=args[args.index('--owner')+1]
p=subprocess.run(['npu-guard','run',*args])
if p.returncode==75:
    receipt.with_name('not_launched.json').write_text(json.dumps({'owner':owner,'guard_returncode':75,'workload_launched':False,'reason':'Host guard reservation deadline reached'}))
    raise SystemExit(75)
g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
if g.get('owner')!=owner:
    raise SystemExit('Guard receipt changed before capture; performance cannot be accepted')
receipt.write_text(json.dumps(g))
raise SystemExit(p.returncode)
