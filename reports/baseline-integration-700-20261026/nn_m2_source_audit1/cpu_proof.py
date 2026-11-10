"""Device-free exact-route and symbolic layout proof. Standard library only."""
import builtins, hashlib, json, os, sys, types
from pathlib import Path
P=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
original_import=builtins.__import__
def guarded_import(name,*args,**kwargs):
 if name.split('.')[0] in ('torch','torch_npu','acl','pyacl') and name not in sys.modules:
  raise AssertionError('Device-capable import forbidden: '+name)
 return original_import(name,*args,**kwargs)
builtins.__import__=guarded_import
m=types.ModuleType('audit_candidate');m.__file__=str(P/'integration700_nn_m2_source_audit1.py')
exec(compile(Path(m.__file__).read_bytes(),m.__file__,'exec'),m.__dict__)
assert m._KEYS==frozenset([('GemmFwdOp','ds-v3-m2-qkv-a-bfloat16')])
class UnsupportedCase(Exception): pass
stub=types.ModuleType('benchmarks.ascend_baselines');stub.UnsupportedCase=UnsupportedCase
sys.modules[stub.__name__]=stub
case=lambda op,cid:types.SimpleNamespace(id=cid,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)),params={'trans_a':False,'trans_b':True})
for op,cid in [('GemmFwdOp','ds-v3-m1-qkv-a-bfloat16'),('GemmW4A16FwdOp','ds-v3-m2-qkv-a-bfloat16')]:
 try:m.make(case(op,cid));raise AssertionError('accepted wrong key')
 except UnsupportedCase:pass
os.environ['OPS_NN_VENDOR']=str(m._VENDOR);os.environ['ASCEND_CUSTOM_OPP_PATH']=str(m._VENDOR)
operations=[]
class Tensor:
 def __init__(self,shape,dtype='bfloat16',origin='x'):self.shape=shape;self.dtype=dtype;self.device='SYMBOLIC';self.origin=origin
 @property
 def T(self):operations.append(('transpose',self.origin));return Tensor(self.shape[::-1],self.dtype,self.origin)
 def float(self):operations.append(('lossless_bf16_to_fp32',self.origin));return Tensor(self.shape,'float32',self.origin)
 def contiguous(self):return self
 def to(self,dtype):operations.append(('output_cast',dtype));return Tensor(self.shape,dtype,'output')
torch=types.ModuleType('torch');torch.float32='float32';torch.empty=lambda shape,device,dtype:Tensor(shape,dtype,'output');sys.modules['torch']=torch
class Graph:
 def __init__(self,library):assert library=='ops-nn'
 def call(self,api,ins,outs,attrs):
  assert api=='MatMulV3' and [x.shape for x in ins[:2]]==[(2,7168),(2112,7168)]
  assert all(x.dtype=='float32' for x in ins[:2]+outs)
  assert ins[2:]==[None,None] and outs[0].shape==(2,2112)
  assert attrs=={'transpose_x1':False,'transpose_x2':True}
  operations.append(('original_complete_api',api))
graph=types.ModuleType('benchmarks.ascend_all_pairs.ops_nn.graph');graph.GraphCall=Graph;sys.modules[graph.__name__]=graph
common=types.ModuleType('benchmarks.ascend_all_pairs.ops_nn.common');common.candidate=lambda b,run,*args:run;sys.modules[common.__name__]=common
run=m.make(case('GemmFwdOp','ds-v3-m2-qkv-a-bfloat16'))
out=run(Tensor((2,7168),origin='a'),Tensor((2112,7168),origin='b'))
assert out.shape==(2,2112) and out.dtype=='bfloat16'
assert sum(x[0]=='original_complete_api' for x in operations)==1
# Execute the exact proposed patch additions with make replaced by sentinel.
patch=(P/'REGISTRATION.patch').read_text();append='\n'.join(x[1:] for x in patch.splitlines() if x.startswith('+') and not x.startswith('+++'))+'\n'
prior=lambda c:'prior';scope={'PROVIDERS':{'GemmFwdOp':prior}}
exec(compile(append,'REGISTRATION.patch','exec'),scope)
scope['_nn_m2'].make=lambda c:'exact'
assert scope['PROVIDERS']['GemmFwdOp'](case('GemmFwdOp','ds-v3-m2-qkv-a-bfloat16'))=='exact'
assert scope['PROVIDERS']['GemmFwdOp'](case('GemmFwdOp','other'))=='prior'
for p in P.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
result={'passed':True,'device_imports':False,'torch':'explicit symbolic stub only','numerical_run':False,'exact_key_and_negative_keys':True,'fallback_identity':True,'single_original_api_symbolic_contract':True,'full_output_shape':[2,2112],'operations':operations,'module_sha256':hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()}
(P/'CPU_PROOF.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
