import os,sys,importlib.abc
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0';os.environ['CUDA_VISIBLE_DEVICES']=''
class BlockDevice(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch_npu','acl','aclnn','tilelang','triton'):raise ImportError('CPU-only test forbids '+fullname)
sys.meta_path.insert(0,BlockDevice())
from pathlib import Path
import ast,json,types
P=Path(__file__).resolve().parent;W=P.parents[2];sys.path.insert(0,str(W/'TileOPs'))
import torch
from workloads.numerics import Custom,zeroed_input
from workloads.device import set_run_device,run_device
set_run_device('cpu')
s=(P/'ascend_pair_check.proposed.py').read_text();tree=ast.parse(s)
funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['verify_fp64_on_cpu','tensor_metadata','observe_call']]
from functools import wraps
ns={"wraps":wraps};exec(compile(ast.Module(body=funcs,type_ignores=[]),'cpu_dispatch_test','exec'),ns)
fn=ns['verify_fp64_on_cpu'];results={}
def validate(got,expected):
 torch.testing.assert_close(got[0],expected[0],rtol=0,atol=0)
 torch.testing.assert_close(got[1],expected[1],rtol=0,atol=0)
def make():
 x=torch.tensor([1.,2.]);h=torch.tensor([3.,4.])
 return types.SimpleNamespace(inputs=(x,h),reference=lambda x,h:(x*2,h+1),verification=Custom(validate,'both outputs',controls=(zeroed_input(0,'query-zeroed'),)))
c=make();r=fn(torch,c,lambda x,h:(x*2,h+1),c.inputs,None,'test',[])
assert r['test'].checked_outputs==r['test'].total_outputs==2;results['full_two_output_and_negative_control']=True
c=make()
try:fn(torch,c,lambda x,h:(x*2,h),c.inputs,None,'test',[])
except AssertionError:results['wrong_second_output_rejected']=True
else:raise AssertionError('wrong second output passed')
c=make()
def mutate(x,h):
 x.add_(1);return x*2,h+1
try:fn(torch,c,mutate,c.inputs,None,'test',[])
except AssertionError:results['candidate_mutation_rejected']=True
else:raise AssertionError('mutating candidate passed')
c=make()
def fail(*args):raise RuntimeError('reference intentionally failed')
c.reference=fail
try:fn(torch,c,lambda x,h:(x*2,h+1),c.inputs,None,'test',[])
except RuntimeError:assert run_device()=='cpu';results['device_restored_on_reference_error']=True
else:raise AssertionError('reference error swallowed')
(P/'CPU_DISPATCH_TESTS.json').write_text(json.dumps(results,indent=2)+'\n');print(results)
