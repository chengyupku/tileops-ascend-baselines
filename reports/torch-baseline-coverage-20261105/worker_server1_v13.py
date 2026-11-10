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
 d=json.loads((P/'SOURCES.v13.server1.json').read_text());assert all(sha(P/n)==h for n,h in d.items()),'frozen source changed';return sha(P/'SOURCES.v13.server1.json')
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
 import torch
 torch.set_num_threads(32 if mode=='prepare' and task['key'][1]=='DSADecodeWithKVCacheFwdOp' else 8)
 from benchmarks.api import cases
 from workloads.device import set_run_device
 from workloads.numerics import compare_outputs,describe
 from tools import kernel_task_profile_full32 as full32
 sys.modules['tools.kernel_task_profile']=full32
 from tools import measure_baselines as core
 core.ROOT=P;core.TILEOPS_ROOT=P/'source'
 set_run_device('cpu');torch.manual_seed(task['seed'])
 case=next(c for c in cases(task['key'][1]) if c.id==task['key'][2])
 assert core._jsonable(case.params)==core._jsonable(task['case']['params']),'exact-case parameters changed'
 def convert(x,device):
  if isinstance(x,torch.Tensor):return x.to(device)
  if isinstance(x,tuple):return tuple(convert(v,device) for v in x)
  if isinstance(x,list):return [convert(v,device) for v in x]
  if isinstance(x,dict):return {k:convert(v,device) for k,v in x.items()}
  return x
 if mode=='prepare':
  xs=case.inputs
  assert case.reference is not None,'missing reference: implement route before enqueue'
  expected=case.reference(*xs)
  put('prepare.json',{'cpu_only':True,'key':task['key'],'params':core._jsonable(case.params),'seed':task['seed'],'sources_sha256':proof,'cpu_seconds':time.time()-started,'oracle':'canonical reference independent CPU execution','verification':describe(case.verification)})
  import shutil,fcntl
  bundle={'inputs':xs,'expected':expected}
  storages={}
  for x in xs:
   if isinstance(x,torch.Tensor):storages[x.untyped_storage().data_ptr()]=x.untyped_storage().nbytes()
  required=sum(storages.values())*2+64*1024**2
  with (P/'handoff-capacity.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   handoff=pathlib.Path('/dev/shm/torch-baseline-coverage');handoff.mkdir(exist_ok=True)
   if shutil.disk_usage(handoff).free<required+1024**3:
    handoff=P/'spool';handoff.mkdir(exist_ok=True)
    assert shutil.disk_usage(handoff).free>required+51*1024**3,'payload would violate disk reserve'
   payload=handoff/(task['id']+'.pt');torch.save(bundle,payload)
  put('payload.json',{'path':str(payload),'sha256':sha(payload),'size':payload.stat().st_size})
  assert not any(n=='torch_npu' or n.startswith('torch_npu.') for n in sys.modules)
  put('PREPARED.json',{'at':time.time(),'seconds':time.time()-started})
 else:
  import torch_npu
  torch.npu.set_device(0);set_run_device('npu')
  torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'})
  torch.npu.matmul.allow_hf32=False
  props=torch.npu.get_device_properties(0);hardware={'device':torch.npu.get_device_name(0),'cube':props.cube_core_num,'vector':props.vector_core_num,'hostname':os.uname().nodename,'torch':torch.__version__,'torch_npu':torch_npu.__version__}
  assert (hardware['cube'],hardware['vector'])==(32,64),hardware
  payload=json.loads((out/'payload.json').read_text());assert sha(payload['path'])==payload['sha256']
  saved=torch.load(payload['path'],map_location='cpu',weights_only=False);pathlib.Path(payload['path']).unlink()
  cpu=saved['inputs'];expected=saved['expected'];case.__dict__['inputs']=cpu
  args=convert(cpu,'npu');fn=case.reference;reset=None;cleanup=None
  if task['key'][1]=='GemmW4A8FwdOp':
   from adapters.w4a8_eager import gemm_w4a8
   def fn(*a):return gemm_w4a8(*a,group_size=case._call.params['group_size'])
   put('device_route.json',{'adapter':'adapters/w4a8_eager.py','formula':'signed INT4, FP8-rounded weight, FP32 contraction, token then channel scale, BF16 output','device_arithmetic':True,'original_cpu_checker':True,'all_operations_timed':True})
  if task['key'][1]=='FFTC2CFwdOp' and args[0].dtype==torch.complex128:
   from adapters.fft_ops_math import bind
   fn,cleanup=bind(tuple(args[0].shape))
   put('device_route.json',{'adapter':'adapters/fft_ops_math.py','origin':'ops-math original aclStft host API, explicit complex unity window','execution':'device AiCPU; not host CPU','device_arithmetic':True,'original_cpu_checker':True})
  before=core._tensor_digest(args)
  produced=fn(*args);torch.npu.synchronize()
  check=compare_outputs(convert(produced,'cpu'),expected,case.verification)
  assert check.checked_outputs and not check.unchecked_reason,dataclasses.asdict(check)
  if getattr(case.verification,'probe',None):case.verification.probe(fn,args)
  torch.npu.synchronize()
  assert before==core._tensor_digest(args),'inputs modified in correctness'
  put('correctness.json',{'status':'verified','check':dataclasses.asdict(check),'oracle':'canonical reference independently executed on CPU','inputs_preserved':True,'policy':describe(case.verification),'source_sha256':proof})
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
