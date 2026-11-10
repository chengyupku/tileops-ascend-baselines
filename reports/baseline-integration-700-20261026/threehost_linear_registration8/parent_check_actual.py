import sys,builtins,json,types,ctypes
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
old=builtins.__import__
def protect(name,*a,**k):
 if name.split('.')[0] in ('torch','torch_npu','acl','tilelang','tvm'):raise AssertionError('device import '+name)
 return old(name,*a,**k)
builtins.__import__=protect
sys.addaudithook(lambda e,a: (_ for _ in ()).throw(AssertionError('dlopen forbidden')) if e=='ctypes.dlopen' else None)
import parent_build_contracts as B
from benchmarks.ascend_all_pairs.registry import candidate_factory
from benchmarks.ascend_baselines import UnsupportedCase
from benchmarks.ascend_all_pairs.ops_transformer import PROVIDERS
cs=B.cases()
for c in cs:
 x=candidate_factory(c,'ops-transformer')();assert callable(x.implementation.run);x.cleanup()
 bad=types.SimpleNamespace(id=c.id,_call=c._call,params=dict(c.params,unexpected=True))
 try:candidate_factory(bad,'ops-transformer')()
 except UnsupportedCase:pass
 else:raise AssertionError('wrong contract accepted')
f=PROVIDERS['DeltaNetFwdOp'];assert f.__module__=='_baseline700_linear8_pinned'
previous=f.previous
unknown=types.SimpleNamespace(id='unknown',_call=cs[0]._call,params=cs[0].params)
if previous is None:
 try:candidate_factory(unknown,'ops-transformer')()
 except UnsupportedCase:pass
 else:raise AssertionError('unknown contract accepted without prior provider')
else:
 for cell in f.__closure__:
  if cell.cell_contents is previous:cell.cell_contents=lambda c:'previous'
 assert candidate_factory(unknown,'ops-transformer')()=='previous'
assert not any(x in sys.modules for x in ('torch','torch_npu','acl','tilelang','tvm'))
print(json.dumps(dict(passed=True,exact_routes=len(cs),wrong_contract_rejection=True,unknown_fallback=True,lazy=True,device_imports=0,NPU_runs=0)))
