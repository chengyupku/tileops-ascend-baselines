import sys,os,pathlib,json,hashlib,time,dataclasses,ctypes,importlib.abc
P=pathlib.Path(__file__).resolve().parent
B=pathlib.Path('/opt/torch-all2208-20261105')
sys.path[:0]=[str(P),str(B/'source/src'),str(B/'source'),str(B),'/opt/tileops-ci/runtime/lib/python3.12/site-packages']
mode=sys.argv[1]; selected=json.loads((P/'TASKS.json').read_text())
if len(sys.argv)>2:selected=[selected[int(sys.argv[2])]]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,default=str))
if mode=='prepare':
 class NoNPU(importlib.abc.MetaPathFinder):
  def find_spec(self,fullname,path=None,target=None):
   if fullname.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU prepare forbids '+fullname,name=fullname)
 sys.meta_path.insert(0,NoNPU())
else:
 import fcntl
 with open('/run/lock/ascend-npu/device-0.lock') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:pass
  else:raise RuntimeError('host guard required')
import torch
from benchmarks.api import cases
from workloads.device import set_run_device
from workloads.numerics import compare_outputs,describe
from tools import kernel_task_profile_full32 as parser
sys.modules['tools.kernel_task_profile']=parser
from tools import measure_baselines as core
import fixes
core.ROOT=B;core.TILEOPS_ROOT=B/'source'
torch.set_num_threads(8)
def convert(x,device):
 if isinstance(x,torch.Tensor):return x.to(device)
 if isinstance(x,tuple):return tuple(convert(v,device) for v in x)
 if isinstance(x,list):return [convert(v,device) for v in x]
 if isinstance(x,dict):return {k:convert(v,device) for k,v in x.items()}
 return x
proof={str(p.relative_to(B)):sha(p) for d in ['source/workloads','source/benchmarks','source/src/tileops','tools'] for p in (B/d).rglob('*.py')}
proof.update({'candidate/'+p.name:sha(p) for p in P.glob('*.py')})
put(P/'SOURCE_PROOF.json',proof)
if mode!='prepare':
 import torch_npu
 torch.npu.set_device(0)
 torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'})
 torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
 props=torch.npu.get_device_properties(0)
 hardware={'device':torch.npu.get_device_name(0),'cube':props.cube_core_num,'vector':props.vector_core_num,'hostname':os.uname().nodename,'torch':torch.__version__,'torch_npu':torch_npu.__version__}
 assert (hardware['cube'],hardware['vector'])==(32,64)
 fd=B/'flush';src=json.loads((fd/'SOURCE.json').read_text())
 assert sha(fd/'host_kernel.asc')==src['generated_source_sha256'] and sha(fd/'flush.aibin')==src['device_binary_sha256']
 dll=ctypes.CDLL(str(fd/'flush.so'));dll.init.argtypes=[ctypes.c_char_p];dll.init.restype=ctypes.c_int;dll.call.argtypes=[ctypes.c_void_p,ctypes.c_void_p];dll.call.restype=ctypes.c_int
 assert dll.init(str(fd/'flush.aibin').encode())==0
 def flush(buffer):
  if dll.call(buffer.data_ptr(),torch.npu.current_stream().npu_stream):raise RuntimeError('flush failed')
 buffer=torch.ones(268435456,dtype=torch.int32,device='npu');flush(buffer);torch.npu.synchronize();assert torch.count_nonzero(buffer).item()==0
 core._flush_kernel_cache[0]=(flush,buffer)
 # Exact truth checks, including both signed zeros, subnormals, infinities and NaNs.
 edge=[]
 for dtype,itype,words in [(torch.complex64,torch.int32,[0,-2147483648,1,-2147483647,2139095040,2143289345]),(torch.complex128,torch.int64,[0,-9223372036854775808,1,-9223372036854775807,9218868437227405312,9221120237041090561])]:
  raw=torch.tensor([(a,b) for a in words for b in words],dtype=itype)
  x=raw.view(dtype).reshape(6,6)
  for y in [x,x.T]:
   got=fixes.complex_truth(y.to('npu')).cpu();want=y.bool();assert torch.equal(got,want)
  edge.append({'dtype':str(dtype),'patterns':72,'exact':True})
 put(P/'EDGE_CHECKS.json',edge)
for r in selected:
 ident=r['op']+'--'+r['case'];out=P/'runs'/ident;out.mkdir(parents=True,exist_ok=True)
 set_run_device('cpu');torch.manual_seed(0)
 case=next(c for c in cases(r['op']) if c.id==r['case'])
 if mode=='prepare':
  xs=case.inputs;expected=case.reference(*tuple(x.clone() if isinstance(x,torch.Tensor) else x for x in xs))
  torch.save({'inputs':xs,'expected':expected},out/'payload.pt')
  put(out/'prepare.json',{'params':core._jsonable(case.params),'verification':describe(case.verification),'seed':0,'payload_sha256':sha(out/'payload.pt'),'original_CPU_reference':True})
  print('PREPARED',ident,flush=True);continue
 set_run_device('npu');saved=torch.load(out/'payload.pt',weights_only=False);case.__dict__['inputs']=saved['inputs'];args=convert(saved['inputs'],'npu')
 params=case._call.params
 fn=(lambda *xs:fixes.int4_quant_per_group(*xs,group_size=params['group_size'])) if r['op']=='INT4QuantPerGroupFwdOp' else (lambda *xs:fixes.reduction(r['op'],*xs,params))
 before=core._tensor_digest(args);got=fn(*args);torch.npu.synchronize()
 check=compare_outputs(convert(got,'cpu'),saved['expected'],case.verification)
 assert check.checked_outputs and not check.unchecked_reason,dataclasses.asdict(check)
 if getattr(case.verification,'probe',None):case.verification.probe(fn,args)
 assert before==core._tensor_digest(args)
 put(out/'correctness.json',{'status':'verified','check':dataclasses.asdict(check),'verification':describe(case.verification),'original_CPU_reference':True,'inputs_preserved':True})
 for variant,call in ([('fixed',fn)] if mode=='probe' else [('original',case.reference),('fixed',fn)]):
  os.environ['TILEOPS_PROFILE_DIR']=str(out/variant/'profiles')
  start=time.time();timing=core._time_reference(torch,torch_npu,call,args,3,1 if mode=='probe' else 15)
  assert before==core._tensor_digest(args)
  put(out/variant/('probe.json' if mode=='probe' else 'measurement.json'),{'key':[r['library'],r['op'],r['case']],'timing':timing,'hardware':hardware,'source_proof_sha256':sha(P/'SOURCE_PROOF.json'),'correctness_sha256':sha(out/'correctness.json'),'seconds':time.time()-start,'old_measurement':r['measurement_id']})
  print('MEASURED',ident,variant,timing['latency_ms'],flush=True)
 torch.npu.synchronize()
