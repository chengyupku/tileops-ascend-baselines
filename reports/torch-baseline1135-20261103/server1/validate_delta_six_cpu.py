import sys,pathlib,json,time,importlib.abc,os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0,NoNPU());p=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(p/'source'),str(p/'source/src'),str(p)]
import torch
from adapters.delta_chunk_six_server1_v1 import reference
from workloads.linear_attention.chunk_cpu import deltanet_chunk_fwd_cpu
from workloads.linear_attention.deltanet import chunkwise_verification
from workloads.numerics import compare_outputs
from workloads.device import set_run_device
from benchmarks.api import cases
set_run_device('cpu');torch.set_num_threads(8);results=[]
for dtype in (torch.float32,torch.float16,torch.bfloat16):
 for C in (1,3,8,64):
  torch.manual_seed(42);q=torch.randn(1,2,2*C,7,dtype=dtype)*.1;k=torch.randn_like(q)*.1;v=torch.randn(1,2,2*C,5,dtype=dtype)*.1;b=torch.rand(1,2,2*C,dtype=dtype)*.5
  check=compare_outputs(reference(q,k,v,b,C),deltanet_chunk_fwd_cpu(q,k,v,b,C),chunkwise_verification(dtype));results.append({'dtype':str(dtype),'chunk':C,'checked':check.checked_outputs})
for row in json.loads((p/'server1/PLAN.json').read_text())['rows']:
 if row['key'][1]!='DeltaNetChunkFwdOp':continue
 torch.manual_seed(row['seed']);case=next(c for c in cases(row['key'][1]) if c.id==row['key'][2]);xs=case.inputs;check=compare_outputs(reference(*xs,64),case.reference(*xs),case.verification);results.append({'key':row['key'],'checked':check.checked_outputs})
(p/'server1/DELTA_SIX_CPU_VALIDATION.json').write_text(json.dumps({'at':time.time(),'checks':results,'device_imports':False},indent=2));print(results)
