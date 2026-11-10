"""Frozen CPU preparation and guarded NPU measurement; no task starts at import."""
import os,sys,pathlib,json,time,hashlib,traceback,dataclasses,importlib.abc
P=pathlib.Path(__file__).resolve().parent
mode,taskfile,outdir=sys.argv[1:];task=json.loads(pathlib.Path(taskfile).read_text());out=pathlib.Path(outdir);out.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(P/'source/src'),str(P/'source'),str(P),'/opt/tileops-ci/runtime/lib/python3.12/site-packages']
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0';os.environ['FLA_NPU_DISABLE_PTH']='1'
def put(name,data):
 f=out/name;q=f.with_suffix('.tmp');q.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str));q.replace(f)
def sha(f):
 with pathlib.Path(f).open('rb') as fp:return hashlib.file_digest(fp,'sha256').hexdigest()
def source_proof():
 d=json.loads((P/'SOURCES.v5.json').read_text());assert all(sha(P/n)==h for n,h in d.items()),'frozen source changed';return sha(P/'SOURCES.v5.json')
if mode=='prepare':
 class NoNPU(importlib.abc.MetaPathFinder):
  def find_spec(self,fullname,path=None,target=None):
   if fullname.split('.')[0] in ('torch_npu','acl'):
    raise ModuleNotFoundError('CPU prepare forbids '+fullname,name=fullname)
 sys.meta_path.insert(0,NoNPU())
elif mode=='device':
 import fcntl
 with open('/run/lock/ascend-npu/device-0.lock') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:pass
  else:raise RuntimeError('device requires held host guard')
else:raise ValueError(mode)
started=time.time()
try:
 proof=source_proof()
 if task['key'][1]=='DeltaNetChunkFwdOp':
  raise RuntimeError('Route audit: current six-output reference computes on CPU; original torch NPU counterpart returns only output o. Requires full-output device counterpart, not NPU-copy timing.')
 import torch
 torch.set_num_threads(32 if mode=='prepare' and task['key'][1]=='DSADecodeWithKVCacheFwdOp' else 8)
 from benchmarks.api import cases
 from workloads.device import set_run_device
 from workloads.numerics import compare_outputs,describe
 from tools import measure_baselines as core
 core.ROOT=P;core.TILEOPS_ROOT=P/'source'
 set_run_device('cpu');torch.manual_seed(task['seed'])
 if task['key'][1]=='GLAFwdOp':
  import workloads.linear_attention.gla as gla
  from adapters import gla_eager,gla_reference
  gla.gla_inference_ref=(gla_reference.reference if mode=='prepare' else gla_eager.reference)
 if task['key'][1]=='DeltaNetFwdOp':
  import workloads.linear_attention.deltanet as delta
  from adapters.delta_reference import reference as delta_ref
  def delta_route(*args,scale=None,l2norm=False):
   return delta_ref(*args,scale=scale,use_qk_l2norm_in_kernel=l2norm)
  delta.deltanet_inference_ref=delta_route
 case=next(c for c in cases(task['key'][1]) if c.id==task['key'][2])
 assert core._jsonable(case.params)==task['case']['params'],'exact-case parameters changed'
 def convert(x,device,clone=False):
  if isinstance(x,torch.Tensor):return x.to(device).clone() if clone else x.to(device)
  if isinstance(x,tuple):return tuple(convert(v,device,clone) for v in x)
  if isinstance(x,list):return [convert(v,device,clone) for v in x]
  if isinstance(x,dict):return {k:convert(v,device,clone) for k,v in x.items()}
  return x
 if mode=='prepare':
  xs=case.inputs;ref=case.reference
  if ref is None:raise RuntimeError('missing reference')
  expected=ref(*convert(xs,'cpu',True))
  put('prepare.json',{'cpu_only':True,'key':task['key'],'params':core._jsonable(case.params),'seed':task['seed'],'sources_sha256':proof,'cpu_seconds':time.time()-started,'oracle':'existing gla_reference FP64 recurrence' if task['key'][1]=='GLAFwdOp' else 'unchanged manifest reference on CPU','verification':describe(case.verification)})
  # Serialize capacity admission across CPU preparers; retain the disk floor.
  import fcntl,shutil
  bundle={'inputs':xs,'expected':expected,'workload_tensors':getattr(case.workload,'tensors',None)};storages={}
  def account(x):
   if isinstance(x,torch.Tensor):
    st=x.untyped_storage();storages[(str(x.device),st.data_ptr())]=st.nbytes()
   elif isinstance(x,dict):
    for v in x.values():account(v)
   elif isinstance(x,(tuple,list)):
    for v in x:account(v)
  account(bundle);required=sum(storages.values())+64*1024**2
  with (P/'handoff-capacity.lock').open('a') as capacity_lock:
   fcntl.flock(capacity_lock,fcntl.LOCK_EX)
   handoff=pathlib.Path('/dev/shm/torch-baseline1135');handoff.mkdir(exist_ok=True)
   if shutil.disk_usage(handoff).free < required+1024**3:
    handoff=P/'spool';handoff.mkdir(exist_ok=True)
    if shutil.disk_usage(handoff).free < required+52*1024**3:
     raise RuntimeError('CPU handoff capacity unavailable: need payload plus 52GiB disk reserve')
   payload=handoff/(task['id']+'.pt');partial=handoff/(task['id']+'.partial')
   try:
    torch.save(bundle,partial);partial.replace(payload)
   except BaseException:
    partial.unlink(missing_ok=True);raise
  put('payload.json',{'path':str(payload),'size':payload.stat().st_size,'sha256':sha(payload)})
  assert not any(n=='torch_npu' or n.startswith('torch_npu.') for n in sys.modules)
  put('PREPARED.json',{'at':time.time(),'seconds':time.time()-started})
 else:
  import torch_npu
  torch.npu.set_device(0);set_run_device('npu')
  if task['key'][1] in ('GemmFwdOp','Conv2dFwdOp'):
   torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'})
   torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
   put('precision_options.json',{'ACL_PRECISION_MODE':'must_keep_origin_dtype','matmul_hf32':torch.npu.matmul.allow_hf32,'conv_hf32':torch.npu.conv.allow_hf32})
  props=torch.npu.get_device_properties(0);hardware={'device':torch.npu.get_device_name(0),'cube':props.cube_core_num,'vector':props.vector_core_num,'hostname':os.uname().nodename,'torch':torch.__version__,'torch_npu':torch_npu.__version__}
  assert (hardware['cube'],hardware['vector'])==(32,64),hardware
  payload=json.loads((out/'payload.json').read_text());assert sha(payload['path'])==payload['sha256'];saved=torch.load(payload['path'],map_location='cpu',weights_only=False);pathlib.Path(payload['path']).unlink()
  cpu=saved['inputs'];expected=saved['expected'];case.__dict__['inputs']=cpu
  args=convert(cpu,'npu')
  if saved.get('workload_tensors') is not None:
   case.workload.tensors=convert(saved['workload_tensors'],'npu')
  fn=case.reference;reset=None
  if task['key'][1] in ('MaxPool2dIndicesFwdOp','MaxPool3dIndicesFwdOp'):
   original_fn=fn
   def fn(*a):
    values,indices=original_fn(*a)
    return values,indices.to(torch.int64)
   put('interface_adaptation.json',{'change':'native pool indices int32 to contract int64','cast_in_timed_call':True})
  if task['key'][1]=='SSDRecurrentFwdOp':
   from benchmarks.baselines import private_inputs
   impl=private_inputs(fn,args,5);args,reset=impl.args,impl.reset
   reset()
  before=core._tensor_digest(args)
  produced=fn(*args);torch.npu.synchronize()
  try:
   check=compare_outputs(convert(produced,'cpu'),expected,case.verification)
  except AssertionError:
   if task['key'][1] in ('GemmFwdOp','Conv2dFwdOp'):
    try:
     promoted=tuple(x.double() if isinstance(x,torch.Tensor) and x.is_floating_point() else x for x in cpu)
     high=case.reference(*promoted).to(expected.dtype)
     diag={}
     for label,value in [('cpu_original',expected),('npu_original',produced.cpu())]:
      try:compare_outputs(value,high,case.verification);diag[label]={'pass':True}
      except AssertionError as error:diag[label]={'pass':False,'error':str(error)}
     put('numeric_diagnostic.json',diag)
    except Exception as error:put('numeric_diagnostic.json',{'diagnostic_error':str(error)})
   raise
  assert check.checked_outputs and not check.unchecked_reason,dataclasses.asdict(check)
  if getattr(case.verification,'probe',None):case.verification.probe(fn,args)
  if reset:reset();torch.npu.synchronize()
  assert before==core._tensor_digest(args),'inputs modified in correctness'
  put('correctness.json',{'status':'verified','check':dataclasses.asdict(check),'oracle':'CPU independent execution of frozen reference','inputs_preserved':True,'policy':describe(case.verification),'source_sha256':proof})
  del produced,expected,saved,cpu
  # The already exported original 1GiB flush kernel; no compiler/schedule changes.
  import ctypes
  fd=P/'flush';src=json.loads((fd/'SOURCE.json').read_text());assert sha(fd/'host_kernel.asc')==src['generated_source_sha256'];assert sha(fd/'flush.aibin')==src['device_binary_sha256']
  dll=ctypes.CDLL(str(fd/'flush.so'));dll.init.argtypes=[ctypes.c_char_p];dll.init.restype=ctypes.c_int;dll.call.argtypes=[ctypes.c_void_p,ctypes.c_void_p];dll.call.restype=ctypes.c_int
  assert dll.init(str(fd/'flush.aibin').encode())==0
  def flush(buffer):
   if dll.call(buffer.data_ptr(),torch.npu.current_stream().npu_stream):raise RuntimeError('flush launch failed')
  buffer=torch.ones(268435456,dtype=torch.int32,device='npu');flush(buffer);torch.npu.synchronize();assert torch.count_nonzero(buffer).item()==0
  core._flush_kernel_cache[0]=(flush,buffer)
  os.environ['TILEOPS_PROFILE_DIR']=str(out/'profiles')
  timing=core._time_reference(torch,torch_npu,fn,args,3,15,reset=reset)
  if reset:reset();torch.npu.synchronize()
  assert before==core._tensor_digest(args),'inputs modified during timing'
  assert proof==source_proof()
  put('measurement.json',{'status':'measured_pending_guard','key':task['key'],'task_id':task['id'],'hardware':hardware,'timing':timing,'correctness_sha256':sha(out/'correctness.json'),'source_sha256':proof,'started_at':started,'finished_at':time.time(),'warmup':3,'repeats':15,'inputs_preserved':True})
  torch.npu.synchronize()
except BaseException as e:
 put(mode+'_failure.json',{'at':time.time(),'key':task['key'],'error':str(e),'traceback':traceback.format_exc()});raise
