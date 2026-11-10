"""Bounded two-round experiment only, never expands a production queue."""
import sys,json,subprocess,time,os,fcntl
from pathlib import Path
from baseline_cache_guard_binding import capture_guard,bind_guard
ROOT=Path(__file__).resolve().parents[1];C=ROOT/'reports/baseline-remote-equivalence-full-20261030';host=sys.argv[1];H=C/host;H.mkdir(exist_ok=True)
lock=(H/'LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
plan=json.loads((C/'PLAN.json').read_text());results=json.loads((H/'RESULTS.json').read_text()) if (H/'RESULTS.json').exists() else []
if results:
 (H/('RESULTS.before-vendor4-'+str(time.time_ns())+'.json')).write_text(json.dumps(results))
 results=[r for r in results if r['status']=='valid']
def put(p,r):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(r,ensure_ascii=False,indent=2));q.replace(p)
identity={'pid':os.getpid(),'start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])}
for rnd in range(2):
 for task in (plan['pairs'] if rnd==0 else list(reversed(plan['pairs']))):
  if any(r['round']==rnd+1 and r['id']==task['id'] and r['status']=='valid' for r in results):continue
  out=H/f'round{rnd+1}'/task['id']
  if out.exists():
   out=out/('retry-workspace-vendor' if (out/'retry-vendor4').exists() else 'retry-vendor4')
  out.mkdir(parents=True,exist_ok=False);put(out/'task.json',task)
  put(H/'STATE.json',{'phase':'running','identity':identity,'round':rnd+1,'key':task['key'],'at':time.time()})
  before=capture_guard();owner=f'baseline-equivalence-full-20261030/{host}/{task["id"]}'
  cmd=['npu-guard','run','--owner',owner,'--wait','600','--timeout','600','--','bash',str(ROOT/'tools/baseline_equivalence_python_fixed.sh'),str(ROOT/'tools/baseline_equivalence_full_entry.py'),'--task',str(out/'task.json'),'--out',str(out)]
  with (out/'device.log').open('w') as log:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
  g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());put(out/'guard.json',g)
  m=json.loads((out/'measurement.json').read_text()) if (out/'measurement.json').exists() else {}
  row={'round':rnd+1,'key':task['key'],'id':task['id'],'exit_code':rc,'measurement':str(out/'measurement.json'),'guard':str(out/'guard.json'),'status':'invalid'}
  if rc==0 and m.get('status')=='measured_pending_guard' and g.get('owner')==owner:
   m.update(guard_receipt=str(out/'guard.json'))
   try:
    m['guard_binding']=bind_guard(m,before,capture_guard());m.update(status='experiment_valid',performance_valid=True);put(out/'measurement.json',m);row['status']='valid'
   except Exception as e:row['error']=str(e)
  import re
  generated=list((out/'compile_debug').rglob('*_kernel.cpp'))
  selected=json.loads((out/'DEPLOYMENT.json').read_text())['vendor'] if (out/'DEPLOYMENT.json').exists() else ''
  headers=[re.findall(r'#include "([^"]+)"',f.read_text())[0] for f in generated]
  provenance=bool(headers) and all(v.startswith(selected+'/') for v in headers)
  row['source_provenance']={'pass':provenance,'vendor':selected,'compiled_sources':headers}
  if row['status']=='valid' and not provenance:row.update(status='invalid',error='Actual compiler source gate failed')
  row['result']={k:m.get(k) for k in ['status','error','actual_inputs','timing','correctness_receipt','performance_valid','cache_key','identity']}
  # Full provenance stays in the attempt; avoid repeated enormous source maps.
  if row['result'].get('identity'):row['result']['identity']={'environment':{k:v for k,v in m['identity']['environment'].items() if k!='loaded_native_sha256'},'protocol':m['identity']['protocol']}
  results.append(row);put(H/'RESULTS.json',results)
  if row['status']!='valid':
   put(H/'STATE.json',{'phase':'experiment_failed_needs_inspection','identity':identity,'at':time.time(),'last':row});raise SystemExit(1)
put(H/'STATE.json',{'phase':'complete','identity':identity,'at':time.time(),'valid_runs':len(results),'formal_rollout':False})
