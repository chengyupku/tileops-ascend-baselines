import os,sys,json,hashlib,importlib.util,functools,time,shutil
from pathlib import Path
from datetime import datetime,timezone
P=Path(sys.argv[1]).resolve();OP=sys.argv[2];CASE_ID=sys.argv[3];TASK=Path(__file__).resolve().parent;W=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_bmm_backend_full1_r2')
def dump(name,data): (P/name).write_text(json.dumps(data,indent=2,default=str)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
P.mkdir(parents=True,exist_ok=True)
(P/'guard_snapshot.json').write_text(Path('/run/lock/ascend-npu/device-0.lock').read_text())
manifest=json.loads((P.parent/'FROZEN.json').read_text());assert all(sha(Path(p))==h for p,h in manifest['sha256'].items()),'frozen source changed'
row=next(r for r in json.loads((TASK/'CASES.json').read_text()) if r['key'][2]==CASE_ID);assert OP==row['key'][1]
sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
sys.path[:0]=[str(W/'TileOPs'),str(W/'TileOPs/src'),str(W),str(TASK)]
from benchmarks.ascend_all_pairs.ops_transformer import PROVIDERS
from adapter import make_candidate
PROVIDERS[OP]=make_candidate
spec=importlib.util.spec_from_file_location('canonical_checker',W/'TileOPs/scripts/ascend_pair_check.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);c.require_guard()
import torch,torch_npu
from benchmarks.api import cases
from resource_gate import capacity
case=next(x for x in cases(OP) if x.id==CASE_ID)
params=json.loads(json.dumps(case.params))
indices=dict(case._call.indices)
assert all(indices[k]==v for k,v in row['manifest_params'].items()),(indices,row['manifest_params'])
dump('PARAMS.json',dict(raw_params=params,actual_call_indices=indices,expected_indices=row['manifest_params']))
host=capacity();free,total=torch.npu.mem_get_info();en=row['envelope']
admission=dict(host=host,hbm_free=free,hbm_total=total,**en,disk_free=shutil.disk_usage(P).free,stage='before_input_allocation',params=params);dump('RESOURCE_ADMISSION.json',admission)
assert host['effective_available_host_bytes']>=en['host_required'] and free>=en['device_required'] and admission['disk_free']>=en['disk_required'],'capacity envelope unavailable'
dump('RUNTIME.json',dict(torch=torch.__version__,torch_npu=torch_npu.__version__,config=torch.__config__.show(),params=params,environment={k:os.environ.get(k) for k in ['ASCEND_CUSTOM_OPP_PATH','ASCEND_CLASSIC_TRANSFORMER_VENDOR','LD_LIBRARY_PATH']}))
captures={};counts={};timings={}
representative=CASE_ID in ['square-256-bfloat16','qwen3-30b-ep8-empty-bfloat16','whisper-conv1-bias-bfloat16']
def save(name,x):
 if not isinstance(x,torch.Tensor):captures[name]={'value':x};return
 v=x.detach().cpu().contiguous();raw=v.view(torch.uint8).numpy().tobytes()
 store=True
 if store:
  digest=hashlib.sha256(raw).hexdigest()
  previous=next((k for k,v in captures.items() if v.get('sha256')==digest and v.get('stored')),None)
  if previous:os.link(P/(previous+'.bin'),P/(name+'.bin'))
  else:(P/(name+'.bin')).write_bytes(raw)
 captures[name]={'sha256':hashlib.sha256(raw).hexdigest(),'shape':list(v.shape),'dtype':str(v.dtype),'original_device':str(x.device),'bytes':len(raw),'stored':store,'byteorder':sys.byteorder,'original_stride':list(x.stride()),'storage_offset':x.storage_offset(),'serialization':'contiguous logical tensor uint8 bytes'}
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

def settings():
 result={'environment':{k:v for k,v in os.environ.items() if k.startswith(('DNNL','ONEDNN','OMP','MKL','TORCH'))},'torch_version':torch.__version__,'git_version':torch.version.git_version,'num_threads':torch.get_num_threads(),'num_interop_threads':torch.get_num_interop_threads(),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}
 for path in ['get_float32_matmul_precision','backends.mkldnn.enabled','backends.mkldnn.deterministic','backends.mkldnn.allow_tf32','backends.mkldnn.matmul.fp32_precision','backends.fp32_precision','backends.cuda.matmul.allow_bf16_reduced_precision_reduction']:
  try:
   obj=torch
   for part in path.split('.'):obj=getattr(obj,part)
   result[path]=obj() if callable(obj) else obj
  except Exception as e:result[path]={'unavailable':str(e)}
 return result
backend_events=[]
original_bmm=torch.bmm
original_backend=torch.backends.mkldnn.enabled
def backend_event(phase,reference=None):
 event=dict(phase=phase,time=time.time(),mkldnn_enabled=torch.backends.mkldnn.enabled,num_threads=torch.get_num_threads(),bmm_repr=repr(torch.bmm),bmm_identity=id(torch.bmm),bmm_unchanged=torch.bmm is original_bmm,native_events=list(native_events) if 'native_events' in globals() else [])
 if reference is not None:
  import inspect
  event.update(reference_repr=repr(reference),reference_source=inspect.getsource(reference),reference_file=inspect.getsourcefile(reference))
 backend_events.append(event);dump('BACKEND_VARIANT.json',dict(setting='torch.backends.mkldnn.enabled=False',original=original_backend,events=backend_events))
 print('BACKEND_STATE',json.dumps(event,default=str),flush=True)
backend_event('before_configuration')
torch.backends.mkldnn.enabled=False
torch.set_num_threads(4)
backend_event('configured')
original_cpu_reference=c.cpu_reference
def instrumented_cpu_reference(torch_arg,reference,inputs):
 @functools.wraps(reference)
 def record(*args):
  dump('CPU_REFERENCE_SETTINGS.json',settings())
  for i,x in enumerate(args):save(f'cpu_reference_input{i}',x)
  print('CPU_REFERENCE_BEGIN',flush=True)
  backend_event('reference_before',reference)
  assert torch.bmm is original_bmm and not torch.backends.mkldnn.enabled
  output=reference(*args)
  backend_event('reference_after',reference)
  print('CPU_REFERENCE_END',flush=True)
  return output
 return original_cpu_reference(torch_arg,record,inputs)
c.cpu_reference=instrumented_cpu_reference

original_observe=c.observe_call;observe_count=0
def capture_observe(torch,run,records):
 global observe_count
 observe_count+=1
 return original_observe(torch,observed('candidate' if observe_count==1 else 'reference',run),records)
c.observe_call=capture_observe
sys.argv=[str(W/'TileOPs/scripts/ascend_pair_check.py'),'--op',OP,'--case-id',CASE_ID,'--library','ops-transformer','--output',str(P/'representative.json')]
exec((TASK/'native_observation.py').read_text(),globals())
dump('PHASE.json',dict(phase='canonical_main',started=time.time()))
try:
 rc=c.main()
finally:
 backend_event('before_restore')
 torch.backends.mkldnn.enabled=original_backend
 backend_event('restored')
assert json.loads((P/'representative.json').read_text())['input_seed']==1868357590
loaded=sorted(set(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/' in line))
dump('EXECUTION.json',{'utc':datetime.now(timezone.utc).isoformat(),'checker_exit_code':rc,'capture_counts':counts,'timings':timings,'frozen_unchanged':all(sha(Path(p))==h for p,h in manifest['sha256'].items()),'loaded_files':loaded,'mapped_host_hashes':{p:sha(Path(p)) for p in loaded if '/private_provider/' in p and Path(p).is_file()}})
raise SystemExit(rc)
