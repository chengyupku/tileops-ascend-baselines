import sys,os,pathlib,json,hashlib,time,dataclasses,ctypes,importlib.abc,traceback,gc
P=pathlib.Path(__file__).resolve().parent; B=pathlib.Path('/opt/torch-all2208-20261105')
sys.path[:0]=[str(P),str(B/'source/src'),str(B/'source'),str(B),'/opt/tileops-ci/runtime/lib/python3.12/site-packages']
mode=sys.argv[1];rows=json.loads((P/'TASKS.json').read_text());idx=int(sys.argv[2]);r=rows[idx]
out=P/'runs'/str(idx);out.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,default=str))
def convert(x,d):
 if isinstance(x,torch.Tensor):return x.to(d)
 if isinstance(x,tuple):return tuple(convert(v,d) for v in x)
 if isinstance(x,list):return [convert(v,d) for v in x]
 if isinstance(x,dict):return {k:convert(v,d) for k,v in x.items()}
 return x
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
import adapters
import native_adapters
core.ROOT=B;core.TILEOPS_ROOT=B/'source';torch.set_num_threads(8)
set_run_device('cpu');torch.manual_seed(0);case=next(c for c in cases(r['op']) if c.id==r['case'])
if mode=='prepare':
 xs=case.inputs;expected=case.reference(*tuple(x.clone() if isinstance(x,torch.Tensor) else x for x in xs))
 # Original negative controls are exercised against the original CPU oracle.
 controls=[]
 for c in getattr(case.verification,'controls',()):
  faulty=c.run(case.reference,tuple(x.clone() if isinstance(x,torch.Tensor) else x for x in xs))
  try:compare_outputs(faulty,expected,case.verification)
  except AssertionError:controls.append({'name':c.name,'rejected':True})
  else:raise AssertionError('negative control unexpectedly accepted '+c.name)
 torch.save({'inputs':xs,'expected':expected},out/'payload.pt')
 put(out/'prepare.json',{'key':[r['op'],r['case']],'params':core._jsonable(case.params),'verification':describe(case.verification),'seed':0,'payload_sha256':sha(out/'payload.pt'),'controls':controls})
 print('PREPARED',idx,r['op'],r['case'],flush=True);sys.exit(0)
variant=sys.argv[3];attempt=sys.argv[4] if len(sys.argv)>4 else 'a1'
vout=out/variant/attempt;vout.mkdir(parents=True,exist_ok=True)
assert not (vout/'result.json').exists(),'immutable attempt exists'
proof={str(p):sha(p) for folder in [B/'source/workloads/quantization',B/'tools',P] for p in folder.glob('*.py')}
proof[str(B/'source/workloads/numerics.py')]=sha(B/'source/workloads/numerics.py')
put(vout/'SOURCE.json',proof)
phase='initialize'
try:
 import torch_npu
 torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'})
 torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
 prop=torch.npu.get_device_properties(0);assert (prop.cube_core_num,prop.vector_core_num)==(32,64)
 hardware={'torch':torch.__version__,'torch_npu':torch_npu.__version__,'cube':prop.cube_core_num,'vector':prop.vector_core_num,'hostname':os.uname().nodename,'backend':('torchair-ge' if variant=='compile_ge' else os.environ.get('TORCHINDUCTOR_NPU_BACKEND'))}
 saved=torch.load(out/'payload.pt',weights_only=False);assert sha(out/'payload.pt')==json.loads((out/'prepare.json').read_text())['payload_sha256']
 case.__dict__['inputs']=saved['inputs'];params=case._call.params;args=convert(saved['inputs'],'npu');set_run_device('npu')
 fn=adapters.factory(r['op'],params,case.reference)
 if variant=='original':
  if r['op']=='INT4QuantPerGroupFwdOp':fn=lambda w:adapters.int4_group(w,params['group_size'])
  else:fn=case.reference
 elif variant=='compile':
  torch._dynamo.config.suppress_errors=False
  fn=torch.compile(fn,backend='inductor',fullgraph=True,dynamic=False)
 elif variant=='compile_ge':
  sys.path.insert(0,'/opt/tileops-ci/runtime/lib/python3.12/site-packages/torch_npu/dynamo')
  import torchair
  fn=torch.compile(fn,backend=torchair.get_npu_backend(),fullgraph=True,dynamic=False)
 elif variant=='native':
  fn=native_adapters.factory(r['op'],params)
  if fn is None:raise NotImplementedError('native API exact dtype/contract candidate absent')
 elif variant!='fixed':raise ValueError(variant)
 phase='correctness';before=core._tensor_digest(args);start=time.time();got=fn(*args);torch.npu.synchronize();first_seconds=time.time()-start
 check=compare_outputs(convert(got,'cpu'),saved['expected'],case.verification)
 assert check.checked_outputs and not check.unchecked_reason,dataclasses.asdict(check)
 if getattr(case.verification,'probe',None):case.verification.probe(fn,args)
 assert before==core._tensor_digest(args)
 del got
 put(vout/'correctness.json',{'check':dataclasses.asdict(check),'verification':describe(case.verification),'first_call_seconds':first_seconds,'inputs_preserved':True})
 if mode=='edge':
  from edge_checks import check as edge_check
  edge=edge_check(r['op'],params,case.verification,variant)
  put(vout/'result.json',{'status':'edge_checked_pending_guard','checks':edge,'key':[r['op'],r['case']],'variant':variant})
  sys.exit(0)
 phase='flush_setup';fd=B/'flush';src=json.loads((fd/'SOURCE.json').read_text())
 assert sha(fd/'host_kernel.asc')==src['generated_source_sha256'] and sha(fd/'flush.aibin')==src['device_binary_sha256']
 dll=ctypes.CDLL(str(fd/'flush.so'));dll.init.argtypes=[ctypes.c_char_p];dll.init.restype=ctypes.c_int;dll.call.argtypes=[ctypes.c_void_p,ctypes.c_void_p];dll.call.restype=ctypes.c_int
 assert dll.init(str(fd/'flush.aibin').encode())==0
 def flush(buffer):
  if dll.call(buffer.data_ptr(),torch.npu.current_stream().npu_stream):raise RuntimeError('flush failed')
 buffer=torch.ones(268435456,dtype=torch.int32,device='npu');flush(buffer);torch.npu.synchronize();assert torch.count_nonzero(buffer).item()==0;core._flush_kernel_cache[0]=(flush,buffer)
 phase='measurement';os.environ['TILEOPS_PROFILE_DIR']=str(vout/'profiles')
 timing=core._time_reference(torch,torch_npu,fn,args,3,1 if mode=='pilot' else 15)
 assert before==core._tensor_digest(args);assert all(sha(p)==s for p,s in proof.items())
 compiled_files={str(p):sha(p) for p in (P/'cache/inductor').rglob('*') if p.is_file() and p.suffix in ('.py','.cpp','.so','.asc','.json')} if variant=='compile' else {}
 put(vout/'COMPILED_ARTIFACTS.json',compiled_files)
 torch.npu.synchronize()
 put(vout/'result.json',{'status':'measured_pending_guard_and_raw_audit','key':[r['op'],r['case']],'variant':variant,'hardware':hardware,'timing':timing,'source_sha256':sha(vout/'SOURCE.json'),'correctness_sha256':sha(vout/'correctness.json'),'first_call_seconds':first_seconds,'dynamo_counters':{k:dict(v) for k,v in torch._dynamo.utils.counters.items()} if variant.startswith('compile') else {},'windows':1 if mode=='pilot' else 15})
 print('MEASURED',idx,variant,timing['latency_ms'],flush=True)
except Exception as e:
 put(vout/'result.json',{'status':'failed','phase':phase,'error':str(e),'traceback':traceback.format_exc(),'key':[r['op'],r['case']],'variant':variant})
 traceback.print_exc();sys.exit(1)
