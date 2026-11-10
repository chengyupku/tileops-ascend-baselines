import sys,types,json,pathlib,hashlib
import torch
assert not hasattr(torch,'npu')
sys.modules['torch_npu']=types.ModuleType('torch_npu') # inert stub; never imports runtime
calls=[]
torch.ops.load_library=lambda p:calls.append({'load_library_stub':p})
def call(*a):
 calls.append({'dispatch':'torch.ops.npu.sgemmv_shrink','shapes':[list(x.shape) for x in a[:-1]],'dtypes':[str(x.dtype) for x in a[:-1]],'slices':a[-1]})
 a[-2].zero_()
torch.ops.npu.sgemmv_shrink=call
torch.npu=types.SimpleNamespace(synchronize=lambda:None)
# Isolate benchmark imports in this CPU-only wiring test.
import benchmarks.ascend_baselines as base
base._owned_native=lambda *args:None
api=types.ModuleType('benchmarks.api');api.Implementation=lambda run:types.SimpleNamespace(run=run);sys.modules['benchmarks.api']=api
import integration700_sgl_linalg36 as adapter
case=types.SimpleNamespace(_call=types.SimpleNamespace(signature=types.SimpleNamespace(name='BmmFwdOp')),params={})
a=torch.zeros((1,128,128),dtype=torch.bfloat16);b=a.clone()
c=adapter.build(case);out=c.implementation.run(a,b)
assert out.shape==(1,128,128) and len([x for x in calls if 'dispatch' in x])==2
print(json.dumps({'cpu_stub_only':True,'numerical_validation':False,'interpreter':sys.executable,'torch':torch.__file__,'source':str(adapter.SOURCE),'adapter_sha256':hashlib.sha256(pathlib.Path(adapter.__file__).read_bytes()).hexdigest(),'calls':calls},indent=2))
