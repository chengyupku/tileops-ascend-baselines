"""CPU semantic checks of new algebra; no NPU module is imported."""
import os,sys,pathlib,types,json,importlib.abc
P=pathlib.Path(__file__).resolve().parent;B=pathlib.Path('/opt/torch-v010-20261106')
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0';os.environ['FLA_NPU_DISABLE_PTH']='1'
sys.path[:0]=[str(P),str(B/'source/src'),str(B/'source'),str(B),'/opt/tileops-ci/runtime/lib/python3.12/site-packages']
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch_npu','acl'):raise RuntimeError('CPU-only test forbids '+fullname)
sys.meta_path.insert(0,NoNPU())
# Pure algebra branches import the name but never call a backend API.
sys.modules['torch_npu']=types.SimpleNamespace()
import torch
from workloads.mamba import ssd_chunk_state_fwd_ref,ssd_state_passing_fwd_ref,selective_scan_ref
from candidates_v6 import bind
S=types.SimpleNamespace
def case(op,p={}):return S(_call=S(signature=S(name=op),params=p),id='cpu-semantics')
def check(g,w):
 if isinstance(g,tuple):
  for a,b in zip(g,w,strict=True):check(a,b)
 else:torch.testing.assert_close(g,w,rtol=1e-4,atol=1e-5)
torch.manual_seed(191);torch.set_num_threads(4);records=[]

for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 x=torch.randn(2,19,2,3).to(dtype);x[:,7]=float('nan')
 offsets=torch.tensor([0,7,19],dtype=torch.int32);indices=torch.tensor([[1,1],[0,1],[1,2],[0,0],[1,0]],dtype=torch.int32);chunk=4
 c=case('MeanPoolingFwdOp',{'chunk_size':chunk});c.workload=S(accum_dtype=torch.float32)
 g=bind(c,(x,offsets,indices),None).execute(x,offsets,indices)
 spans=[(11,15),(4,7),(15,19),(0,4),(7,11)]
 expected=torch.stack([x[:,a:b].float().mean(1).to(dtype) for a,b in spans],1)
 torch.testing.assert_close(g,expected,rtol=0,atol=0,equal_nan=True)
 records.append(str(dtype)+' ragged tails, reordered chunks, neighbouring NaN')
(P/'CPU_MEAN_CHECK.json').write_text(json.dumps({'passed':records,'device_execution':False},indent=2));print('CPU mean checks passed',len(records))
