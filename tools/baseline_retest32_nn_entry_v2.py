"""Select previously verified top-level ARM NN interface; unchanged checker."""
from pathlib import Path
import sys,json,os
ROOT=Path(__file__).resolve().parents[1]
task=json.loads(Path(sys.argv[sys.argv.index('--task')+1]).read_text())
op=task['key'][1]
# Promote only an interface whose explicit repair already produced a valid cache.
H=ROOT/'reports/baseline-retest32-20261030/server2'
repairs=json.loads((H/'RETRIES.json').read_text()) if (H/'RETRIES.json').exists() else {}
latest={}
if (H/'journal.jsonl').exists():
 for line in (H/'journal.jsonl').read_text().splitlines():
  try:
   r=json.loads(line);latest[r['id']]=r
  except ValueError:pass
for key,r in latest.items():
 repair=repairs.get(key,{})
 if r['status']=='cached' and r['key'][:2]==task['key'][:2] and r.get('revision')==repair.get('revision') and repair.get('entry') in ('tools/baseline_retest32_nn950_entry.py','tools/baseline_retest32_nn950_extra_entry.py','tools/baseline_retest32_nn950_extra_v3_entry.py'):
  import runpy
  runpy.run_path(str(ROOT/repair['entry']),run_name='__main__')
  raise SystemExit(0)
if op in ('MulFwdOp','MinimumFwdOp','MaximumFwdOp','DivFwdOp','PowFwdOp'):
    entry='baseline_old1076_nn_acl_list_v7.py'
elif op=='PreluFwdOp':entry='baseline_old1076_nn_arm_prelu_v4.py'
else:entry='baseline_old1076_nn_arm_staging_v3.py'
# Existing wrappers ultimately execute runpy on the common measure entry.
# Redirect only that entry to the same checker/timing with 32-core identity.
import runpy
original=runpy.run_path
def run_path(path,*args,**kwargs):
    if str(path)==str(ROOT/'tools/measure_baseline_cache_pair.py'):
        path=str(ROOT/'tools/measure_baseline_retest32.py')
    return original(path,*args,**kwargs)
runpy.run_path=run_path
original(str(ROOT/'tools'/entry),run_name='__main__')
