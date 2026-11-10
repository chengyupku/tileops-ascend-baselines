"""Real on-disk registry/factories, CPU-only constructor stubs; NOT numerical verification."""
import sys,os,json,hashlib,importlib,ctypes
from pathlib import Path
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
vendor='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_dense/gemm/private_provider'
os.environ['ASCEND_CUSTOM_OPP_PATH']=vendor
os.environ['ASCEND_CLASSIC_TRANSFORMER_VENDOR']=vendor
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
class NoDevice:
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='torch_npu' or fullname.startswith('torch_npu.') or fullname in ('acl','pyacl'):
   raise RuntimeError('CPU registry proof forbids device backend import: '+fullname)
sys.meta_path.insert(0,NoDevice())
W=Path('/opt/tileops-all-pairs/workspace');P=W/'reports/baseline-integration-700-20261026/gmm_wgrad_first1'
sys.path[:0]=[str(W/'TileOPs'),str(W/'TileOPs/src'),str(W),str(W/'.cache/all-pairs/remote_triton_runtime/overlay/site-packages'),'/opt/tileops-ci/runtime/lib/python3.12/site-packages']
import torch
# Explicitly reject device allocation and backend access; CPU factory creation needs none.
def forbidden(*a,**k):raise RuntimeError('No device action allowed in CPU registry proof')
if hasattr(torch,'npu'):raise AssertionError('Unexpected NPU backend present')
original_cdll=ctypes.CDLL
def guarded_cdll(name,*a,**k):
 if any(s in str(name).lower() for s in ('ascend','cust_op','acl','optiling')):raise RuntimeError('CPU proof forbids device DSO load '+str(name))
 return original_cdll(name,*a,**k)
ctypes.CDLL=guarded_cdll
import benchmarks.ascend_math_more as mathmore
calls=[]
class Fn:
 def __call__(self,*a,**k):return forbidden(*a,**k)
class Base:
 def __init__(self):self.aclCreateTensorList=Fn();self.aclDestroyTensorList=Fn()
class ConstructorStub:
 def __init__(self,api,args,**kw):
  assert os.environ['ASCEND_CLASSIC_TRANSFORMER_VENDOR'] in os.environ['ASCEND_CUSTOM_OPP_PATH'].split(':')
  assert str(kw['host_library'])==vendor+'/op_api/lib/libcust_opapi.so'
  self.base=Base();calls.append({'api':api,'argument_count':len(args),'library':kw['library'],'host_library':str(kw['host_library'])})
 def __getattr__(self,n):return forbidden
mathmore._MathCall=ConstructorStub
from benchmarks.api import cases
from benchmarks.ascend_all_pairs.registry import candidate_factory
pkg=importlib.import_module('benchmarks.ascend_all_pairs.ops_transformer')
proofs=[P/'REGISTRATION_PROOF.json']+sorted(P.glob('*/REGISTRATION_PROOF.json'));rows=[]
for f in proofs:
 d=json.loads(f.read_text());library,op,cid=d['key'];case=next(c for c in cases(op) if c.id==cid)
 before=len(calls);candidate=candidate_factory(case,library)()
 assert len(calls)==before+1 and calls[-1]['api']=='aclnnGroupedMatmulV5'
 assert callable(candidate.implementation.run)
 source=candidate.source_sha256
 expected=d['tested_binding_sha256'];assert expected in source.values(),(cid,expected)
 rows.append({'key':d['key'],'factory_expected':d['factory'],'actual_run_module':candidate.implementation.run.__module__,'actual_run_qualname':candidate.implementation.run.__qualname__,'api':candidate.api,'constructor':calls[-1],'tested_binding_hash_in_actual_candidate':True,'actual_source_sha256':source,'numeric_run_executed':False})
quotas={}
for f in ['/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/cpu/cpu.cfs_quota_us','/sys/fs/cgroup/cpu/cpu.cfs_period_us','/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us','/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us']:
 if Path(f).exists():quotas[f]=Path(f).read_text().strip()
out={'required_tested_vendor_environment':{k:os.environ[k] for k in ('ASCEND_CUSTOM_OPP_PATH','ASCEND_CLASSIC_TRANSFORMER_VENDOR')},'cpu_only':True,'numeric_validation':False,'runtime_constructor_stub':'benchmarks.ascend_math_more._MathCall only; full real registry and actual factory executed; run callable NOT called','torch_npu_imported':'torch_npu' in sys.modules,'thread_environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS']},'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),'cpu_affinity_count':len(os.sched_getaffinity(0)),'cpu_quota':quotas,'registry_path':pkg.__file__,'registry_sha256':hashlib.sha256(Path(pkg.__file__).read_bytes()).hexdigest(),'rows':rows}
(P/'REAL_REGISTRY_CPU_PROOF.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='rows'}));print('actual factories checked',len(rows))
