"""Read-only SSH verification: streamed SHA256 only, no native imports."""
import json,subprocess,shlex
from pathlib import Path
P=Path(__file__).resolve().parent
pins=json.loads((P/'RUNTIME_PINS.json').read_text())
script='''import hashlib,json,platform
from pathlib import Path
pins=json.loads(%r)
checked=0;bad=[]
for group in pins.values():
 for name,digest in group.items():
  h=hashlib.sha256()
  try:
   with open(name,'rb') as f:
    for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
   if h.hexdigest()!=digest:bad.append(name)
  except OSError as e:bad.append([name,str(e)])
  checked+=1
root=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_dense48_r2/vendor')
actual={str(p) for p in root.rglob('*') if p.is_file() and '/ascendc/' not in str(p)}
expected=set(pins['vendor'])
print(json.dumps(dict(status='passed' if not bad and actual==expected else 'failed',checked=checked,mismatches=bad,extra=sorted(actual-expected),missing=sorted(expected-actual),machine=platform.machine(),hostname=platform.node(),native_imports=0,NPU_runs=0)))
'''%json.dumps(pins)
result=subprocess.run(['/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh','openlibing-DevEnv-661495','python3 -c '+shlex.quote(script)],capture_output=True,text=True)
if result.returncode:raise RuntimeError(result.stderr)
x=json.loads(result.stdout);(P/'REMOTE_CPU_CHECK.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
assert x['status']=='passed'
