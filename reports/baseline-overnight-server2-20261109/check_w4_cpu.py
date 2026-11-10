import os,sys,importlib.abc,json,pathlib
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0';os.environ['FLA_NPU_DISABLE_PTH']='1'
P=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(P),'/opt/torch-all2208-20261105/source','/opt/torch-all2208-20261105/source/src','/opt/tileops-ci/runtime/lib/python3.12/site-packages']
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError(fullname,name=fullname)
sys.meta_path.insert(0,NoNPU())
import torch
from candidate_w4 import decode
from workloads.gemm import dequantize_w4a16_weight
rows=[]
for n,kp,g in [(3,64,1),(2,128,4),(4,256,8)]:
 for kind in ('allbytes','random'):
  packed=(torch.arange(n*kp)%256).to(torch.uint8).reshape(n,kp) if kind=='allbytes' else torch.randint(0,256,(n,kp),dtype=torch.uint8)
  scale=torch.rand(n,g,dtype=torch.float16);zero=torch.randint(0,16,(n,g),dtype=torch.uint8)
  expected=dequantize_w4a16_weight(packed,scale,zero);actual=decode(packed,scale,zero)
  assert torch.equal(expected,actual)
  rows.append({'n':n,'kp':kp,'groups':g,'kind':kind,'bit_exact':True})
(P/'W4_CPU_CHECK.json').write_text(json.dumps(rows,indent=2));print('passed',len(rows))
