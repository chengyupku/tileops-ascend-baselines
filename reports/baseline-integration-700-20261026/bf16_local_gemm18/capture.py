import os,sys,json,hashlib,importlib.util,functools
from pathlib import Path
from datetime import datetime,timezone
P=Path(sys.argv[1]).resolve()
CASE_ID=sys.argv[2]
TASK=Path(__file__).resolve().parent
W=Path('/root/dyq/workspace')
D=TASK
def dump(name,data): (P/name).write_text(json.dumps(data,indent=2,default=str)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
(P/'guard_snapshot.json').write_text(Path('/run/lock/ascend-npu/device-0.lock').read_text())
manifest=json.loads((P/'FROZEN.json').read_text())
assert all(sha(Path(p))==h for p,h in manifest['sha256'].items()),'frozen source changed'
sys.path[:0]=[str(W/'TileOPs'),str(W/'TileOPs/src'),str(W),str(D)]

from benchmarks.ascend_all_pairs.ops_transformer import PROVIDERS
from binding import dense_fp32 as dense
PROVIDERS['GemmFwdOp']=dense
spec=importlib.util.spec_from_file_location('canonical_checker',W/'TileOPs/scripts/ascend_pair_check.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
c.require_guard()
import torch,torch_npu,shutil
from benchmarks.api import cases
from resource_gate import capacity
case=next(x for x in cases('GemmFwdOp') if x.id==CASE_ID)
q=case._call.indices
m,n,k=q['M'],q['N'],q['K']
elements=m*k+k*n+m*n
host=capacity();free,total=torch.npu.mem_get_info()
host_required=2*1024**3+elements*32
device_required=2*1024**3+elements*24
disk_required=1024**3+elements*16
admission=dict(host=host,hbm_free=free,hbm_total=total,host_required=host_required,device_required=device_required,disk_required=disk_required,disk_free=shutil.disk_usage(P).free,stage='before_input_allocation',dimensions=[m,n,k])
dump('RESOURCE_ADMISSION.json',admission)
assert host['effective_available_host_bytes']>=host_required and free>=device_required and admission['disk_free']>=disk_required,'capacity envelope unavailable'
dump('RUNTIME.json',dict(torch=torch.__version__,torch_npu=torch_npu.__version__,config=torch.__config__.show()))
captures={}; counts={}
def save(name,x):
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
sys.argv=[str(W/'TileOPs/scripts/ascend_pair_check.py'),'--op','GemmFwdOp','--case-id',CASE_ID,'--library','ops-transformer','--output',str(P/'representative.json')]
rc=c.main()
dump('EXECUTION.json',{'utc':datetime.now(timezone.utc).isoformat(),'checker_exit_code':rc,'capture_counts':counts,'frozen_unchanged':all(sha(Path(p))==h for p,h in manifest['sha256'].items()),'loaded_files':[line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/' in line]})
raise SystemExit(rc)
