import subprocess,json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';host='openlibing-DevEnv-661495'
R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/'
for name in ['DIAGNOSTIC.json','WINDOW_VALUES.json']:
 raw=subprocess.check_output([ssh,host,'cat '+R+P.name+'/'+name]);(P/name).write_bytes(raw)
code=r'''
import json,hashlib
from pathlib import Path
P=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_bmm_input_backend1')
sealed=json.loads((P/'SEALED_HASHES.json').read_text())
out={'sealed_schema':list(sealed)[:8], 'files':{},'tolerance_source':[]}
for f in ['default.bin','disabled.bin','captured_candidate_window.bin','captured_reference_window.bin']:
 p=P/f;out['files'][f]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'sealed_occurrences':[]}
 def walk(v,path=''):
  if isinstance(v,dict):
   for k,x in v.items():
    if f in str(k):out['files'][f]['sealed_occurrences'].append([path+'/'+k,x])
    walk(x,path+'/'+str(k))
  elif isinstance(v,list):
   for i,x in enumerate(v):walk(x,path+'/'+str(i))
 walk(sealed)
for p in [P/'TileOPs/workloads/gemm.py',P/'TileOPs/benchmarks/verification.py',P/'TileOPs/benchmarks/api.py']:
 if not p.exists():continue
 lines=p.read_text().splitlines()
 for i,l in enumerate(lines):
  if any(x in l for x in ['0.016','1.6e-2','bfloat16','tolerance']):out['tolerance_source'].append({'path':str(p),'line':i+1,'text':'\n'.join(lines[max(0,i-2):i+4])})
print(json.dumps(out,indent=2))
'''
r=subprocess.check_output([ssh,host,'/usr/bin/python3 -I -S -B -'],input=code.encode());(P/'ARTIFACT_VERIFICATION.json').write_bytes(r)
print(r.decode()[:6500])
