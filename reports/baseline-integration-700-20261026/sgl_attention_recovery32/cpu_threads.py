import sys,importlib.abc,hashlib,json,time
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]=='torch_npu':raise RuntimeError('torch_npu prohibited in CPU check')
sys.meta_path.insert(0,Block())
import torch
assert 'torch_npu' not in sys.modules
seed=196355653;torch.manual_seed(seed);a=torch.randn(1048576,dtype=torch.float32);b=a.clone();ix=torch.randperm(131072)
start=time.monotonic();a,b=a.to(dtype=torch.float64,device='cpu'),b.to(dtype=torch.float64,device='cpu');finite=torch.isfinite(a)&torch.isfinite(b);error=(a[finite]-b[finite]).abs().max().item()
print(json.dumps({'cpu_only':True,'torch':torch.__file__,'threads':torch.get_num_threads(),'seed':seed,'a_hash':hashlib.sha256(a.numpy().tobytes()).hexdigest(),'randperm_hash':hashlib.sha256(ix.numpy().tobytes()).hexdigest(),'max_abs_err':error,'diagnostic_wall_s':time.monotonic()-start,'torch_npu_imported':'torch_npu' in sys.modules}))
