import os,sys,json,hashlib,importlib.util,functools,time,shutil
from pathlib import Path
from datetime import datetime,timezone
P=Path(sys.argv[1]).resolve();OP=sys.argv[2];CASE_ID=sys.argv[3];TASK=Path(__file__).resolve().parent;W=TASK.parents[2]; mode=sys.argv[4]
def dump(name,data): (P/name).write_text(json.dumps(data,indent=2,default=str)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
P.mkdir(parents=True,exist_ok=True)
(P/'guard_snapshot.json').write_text(Path('/run/lock/ascend-npu/device-0.lock').read_text())
manifest=json.loads((TASK/('FROZEN_'+mode+'.json')).read_text());assert all(sha(Path(p))==h for p,h in manifest['sha256'].items()),'frozen source changed'
row=next(r for r in json.loads((TASK/'CASES.json').read_text()) if r['key'][2]==CASE_ID) if mode=='bf16' else {'manifest_params':{},'envelope':{'host_required':4*2**30,'device_required':4*2**30,'disk_required':2**30}}
sys.path[:0]=[str(W/'TileOPs'),str(W/'TileOPs/src'),str(W),str(TASK)]
from benchmarks.ascend_all_pairs.ops_transformer import PROVIDERS
if mode=='bf16':
 spec=importlib.util.spec_from_file_location('original_adapter',TASK.parent/'bf16_moe_hadamard_native10/adapter.py');ad=importlib.util.module_from_spec(spec);spec.loader.exec_module(ad);make_candidate=ad.make_candidate
else:
 sys.path.insert(0,str(TASK.parent/'transformer_dense'))
 from integration700_transformer_dense_fp8_bmm import fp8_bmm as make_candidate
PROVIDERS[OP]=make_candidate
spec=importlib.util.spec_from_file_location('canonical_checker',W/'TileOPs/scripts/ascend_pair_check.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);c.require_guard()
import torch,torch_npu
from benchmarks.api import cases
from resource_gate import capacity
case=next(x for x in cases(OP) if x.id==CASE_ID)
params=json.loads(json.dumps(case.params,default=str))
indices=dict(case._call.indices)
assert all(json.loads(json.dumps(indices[k],default=str))==v for k,v in row['manifest_params'].items()),(indices,row['manifest_params'])
dump('PARAMS.json',dict(raw_params=params,actual_call_indices=indices,expected_indices=row['manifest_params']))
host=capacity();free,total=torch.npu.mem_get_info();en=row['envelope']
admission=dict(host=host,hbm_free=free,hbm_total=total,**en,disk_free=shutil.disk_usage(P).free,stage='before_input_allocation',params=params);dump('RESOURCE_ADMISSION.json',admission)
assert host['effective_available_host_bytes']>=en['host_required'] and free>=en['device_required'] and admission['disk_free']>=en['disk_required'],'capacity envelope unavailable'
dump('RUNTIME.json',dict(torch=torch.__version__,torch_npu=torch_npu.__version__,config=torch.__config__.show(),params=params,environment={k:os.environ.get(k) for k in ['ASCEND_CUSTOM_OPP_PATH','ASCEND_CLASSIC_TRANSFORMER_VENDOR','LD_LIBRARY_PATH']}))
captures={};counts={};timings={}
representative=CASE_ID in ['square-256-bfloat16','qwen3-30b-ep8-empty-bfloat16','whisper-conv1-bias-bfloat16']
def save(name,x):
 if not isinstance(x,torch.Tensor):captures[name]={'value':x};return
 if 'input' in name:
  captures[name]={'shape':list(x.shape),'dtype':str(x.dtype),'device':str(x.device),'stride':list(x.stride()),'preservation':'original full checker inputs_preserved'};dump('CAPTURE.json',captures);return
 v=x.detach().cpu().contiguous();raw=v.view(torch.uint8).numpy().tobytes()
 store=True
 if store:(P/(name+'.bin')).write_bytes(raw)
 captures[name]={'sha256':hashlib.sha256(raw).hexdigest(),'shape':list(v.shape),'dtype':str(v.dtype),'original_device':str(x.device),'bytes':len(raw),'stored':store}
 if v.numel()<=32:captures[name]['values']=v.tolist()
 dump('CAPTURE.json',captures)
def observed(label,fn):
 @functools.wraps(fn)
 def call(*args,**kw):
  i=counts.get(label,0);counts[label]=i+1;prefix=f'{label}_{i}'
  for j,t in enumerate(args):save(f'{prefix}_input{j}',t)
  dump('PHASE.json',dict(phase=label,started=time.time()))
  start=time.monotonic();out=fn(*args,**kw);torch.npu.synchronize();timings[prefix]=time.monotonic()-start;dump('TIMINGS.json',timings)
  save(f'{prefix}_output',out)
  for j,t in enumerate(args):save(f'{prefix}_post_input{j}',t)
  return out
 return call
original_observe=c.observe_call;observe_count=0
def capture_observe(torch,run,records):
 global observe_count
 observe_count+=1
 return original_observe(torch,observed('candidate' if observe_count==1 else 'reference',run),records)
c.observe_call=capture_observe
sys.argv=[str(W/'TileOPs/scripts/ascend_pair_check.py'),'--op',OP,'--case-id',CASE_ID,'--library','ops-transformer','--output',str(P/'representative.json')]
exec((TASK/'native_observation.py').read_text(),globals())
dump('PHASE.json',dict(phase='canonical_main',started=time.time()))
rc=c.main()
torch.npu.synchronize()
loaded=sorted(set(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/' in line))
dump('EXECUTION.json',{'utc':datetime.now(timezone.utc).isoformat(),'checker_exit_code':rc,'capture_counts':counts,'timings':timings,'frozen_unchanged':all(sha(Path(p))==h for p,h in manifest['sha256'].items()),'loaded_files':loaded,'mapped_host_hashes':{p:sha(Path(p)) for p in loaded if '/bf16gmm_transformer/' in p and Path(p).is_file()}})
raise SystemExit(rc)
