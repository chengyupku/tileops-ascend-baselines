import os,sys,json,hashlib,importlib.util,functools
from pathlib import Path
from datetime import datetime,timezone
P=Path(__file__).resolve().parent
W=Path('/opt/tileops-all-pairs/workspace')
D=W/'reports/baseline-integration-700-20261026/transformer_dense'
def dump(name,data): (P/name).write_text(json.dumps(data,indent=2,default=str)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
(P/'guard_snapshot.json').write_text(Path('/run/lock/ascend-npu/device-0.lock').read_text())
manifest=json.loads((P/'FROZEN.json').read_text())
assert all(sha(Path(p))==h for p,h in manifest['sha256'].items()),'frozen source changed'
sys.path[:0]=[str(W/'TileOPs'),str(W/'TileOPs/src'),str(W),str(D)]
sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
from benchmarks.ascend_all_pairs.ops_transformer import PROVIDERS
sys.path.insert(0,str(P))
from remaining_binding import remaining as dense
PROVIDERS['GroupedGemmFwdOp']=dense
spec=importlib.util.spec_from_file_location('canonical_checker',W/'TileOPs/scripts/ascend_pair_check.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
c.require_guard()
import torch
captures={}; counts={}
def save(name,x):
 if x is None:
  captures[name]={"value":None};dump("CAPTURE.json",captures);return
 v=x.detach().cpu().contiguous(); raw=v.view(torch.uint8).numpy().tobytes()
 (P/(name+'.bin')).write_bytes(raw)
 captures[name]={'sha256':hashlib.sha256(raw).hexdigest(),'shape':list(v.shape),'dtype':str(v.dtype),'original_device':str(x.device),'bytes':len(raw)}
 dump('CAPTURE.json',captures)
def observed(label,fn):
 @functools.wraps(fn)
 def call(*args,**kw):
  i=counts.get(label,0);counts[label]=i+1;prefix=f'{label}_{i}'
  for j,t in enumerate(args):save(f'{prefix}_input{j}',t)
  out=fn(*args,**kw)
  save(f'{prefix}_output',out)
  for j,t in enumerate(args):save(f'{prefix}_post_input{j}',t)
  return out
 return call
original_cpu_reference=c.cpu_reference
def capture_cpu_reference(torch,reference,inputs):
 return original_cpu_reference(torch,observed('reference',reference),inputs)
c.cpu_reference=capture_cpu_reference
original_observe=c.observe_call
observe_count=0
def capture_observe(torch,run,records):
 global observe_count
 observe_count+=1
 if observe_count==1:run=observed('candidate',run)
 return original_observe(torch,run,records)
c.observe_call=capture_observe
sys.argv=[str(W/'TileOPs/scripts/ascend_pair_check.py'),'--op','GroupedGemmFwdOp','--case-id','moe-wgrad-float16','--library','ops-transformer','--output',str(P/'representative.json')]
rc=c.main()
dump('EXECUTION.json',{'utc':datetime.now(timezone.utc).isoformat(),'checker_exit_code':rc,'capture_counts':counts,'frozen_unchanged':all(sha(Path(p))==h for p,h in manifest['sha256'].items()),'loaded_files':[line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/' in line]})
raise SystemExit(rc)
