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
from candidates_v2 import bind,prefix_product
S=types.SimpleNamespace
def case(op,p={}):return S(_call=S(signature=S(name=op),params=p),id='cpu-semantics')
def check(g,w):
 if isinstance(g,tuple):
  for a,b in zip(g,w,strict=True):check(a,b)
 else:torch.testing.assert_close(g,w,rtol=1e-4,atol=1e-5)
torch.manual_seed(191);torch.set_num_threads(4);records=[]
for initial in [None,torch.randn(2,4,8)]:
 x=torch.randn(2,17,4,8);dA=-torch.rand(2,4,17);args=(x,dA,initial)
 got=bind(case('SSDStatePassingFwdOp'),args,None).execute(*args);check(got,ssd_state_passing_fwd_ref(*args));records.append('statepassing initial='+str(initial is not None))
for segmented in [False,True]:
 x=torch.randn(2,64,4,8);b=torch.randn(2,64,2,4);dt=torch.rand(2,4,4,16);da=-dt.cumsum(-1)
 ids=torch.arange(64)[None].expand(2,-1)//7 if segmented else None
 args=(x,b,dt,da,ids);check(bind(case('SSDChunkStateFwdOp'),args,None).execute(*args),ssd_chunk_state_fwd_ref(x,b,dt,da,2,ids));records.append('chunkstate segmented='+str(segmented))
for segmented in [False,True]:
 for initial in [None,torch.randn(2,4,3)]:
  u=torch.randn(2,4,133);dt=torch.randn_like(u);a=-torch.rand(4,3);b=torch.randn(2,2,3,133);c=torch.randn_like(b);d=torch.randn(4);z=torch.randn_like(u);bias=torch.randn(4);ids=torch.arange(133)[None].expand(2,-1)//19 if segmented else None
  args=(u,dt,a,b,c,d,z,bias,ids,initial);check(bind(case('SelectiveScanFwdOp',{'delta_softplus':True}),args,None).execute(*args),selective_scan_ref(*args));records.append('scan segmented='+str(segmented)+' initial='+str(initial is not None))
for x in [torch.tensor([[1.,-2.,0.,-3.,4.,-0.,8.]]),torch.rand(3,127)*.04+.98,torch.tensor([[1.,float('inf'),0.,2.]])]:
 torch.testing.assert_close(prefix_product(x),torch.cumprod(x,-1),atol=1e-5,rtol=1e-4,equal_nan=True);records.append('cumprod shape='+str(list(x.shape)))
(P/'CPU_ALGEBRA_CHECK.json').write_text(json.dumps({'passed':records,'device_execution':False},indent=2));print('CPU algebra checks',len(records),'passed')
