"""CPU-only counterexamples for tempting rewrites; original checker retained."""
import os,sys,importlib.abc,pathlib,json
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError(fullname,name=fullname)
sys.meta_path.insert(0,NoNPU())
import torch
def tree(x):
 y=x.float().clone();stride=1
 while stride<y.shape[-1]:
  y=torch.cat((y[...,:stride],y[...,stride:]*y[...,:-stride]),dim=-1);stride*=2
 return y.to(x.dtype)
rows=[]
for values in [[1e-30,1e30,1e30,1e-30],[1e30,1e-30,1e-30,1e30],[0.,float('inf'),1.,-1.],[-1.,2.,-3.,0.]]:
 x=torch.tensor(values,dtype=torch.float32);expected=torch.cumprod(x,dim=-1);got=tree(x)
 rows.append({'values':values,'original':expected.tolist(),'tree':got.tolist(),'matches_original_bound':torch.allclose(expected,got,atol=1e-3,rtol=1e-3,equal_nan=True)})
p=pathlib.Path(__file__).parent/'CUMPROD_CPU_REVIEW.json';p.write_text(json.dumps({'cpu_only':True,'conclusion':'Naive reassociation is not a general equivalent replacement when intermediate overflow or underflow occurs. Retain original baseline pending a numerically qualified scan implementation.','examples':rows},indent=2));print(p.read_text())
