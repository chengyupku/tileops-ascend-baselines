import sys,builtins,types,json,copy
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));old=builtins.__import__
def protect(name,*a,**k):
 if name.split('.')[0] in ('torch','torch_npu','acl','tilelang','tvm'):raise AssertionError('device import '+name)
 return old(name,*a,**k)
builtins.__import__=protect
sys.addaudithook(lambda e,a: (_ for _ in ()).throw(AssertionError('dlopen')) if e=='ctypes.dlopen' else None)
from build_contracts import cases
from benchmarks.ascend_all_pairs.registry import candidate_factory
from benchmarks.ascend_all_pairs.ops_transformer import PROVIDERS
from benchmarks.ascend_baselines import UnsupportedCase
cs=cases()
for c in cs:
 x=candidate_factory(c,'ops-transformer')();assert callable(x.implementation.run);x.cleanup()
 for k in c._call.params:
  bad=copy.deepcopy(c);bad._call.params[k]='invalid'
  try:candidate_factory(bad,'ops-transformer')()
  except UnsupportedCase:pass
  else:raise AssertionError('wrong contract accepted')
f=PROVIDERS['DeltaNetFwdOp'];assert f.__module__=='_baseline700_linear3_pinned'
assert len(f.__closure__)==1;f.__closure__[0].cell_contents=lambda c:'previous'
c=cs[0];unknown=types.SimpleNamespace(id='unknown',_call=c._call,params=c.params)
assert candidate_factory(unknown,'ops-transformer')()=='previous'
assert not any(n in sys.modules for n in ('torch','torch_npu','acl','tilelang','tvm'))
print(json.dumps(dict(passed=True,exact_routes=3,wrong_contract_rejection=True,unknown_fallback=True,lazy=True,NPU_runs=0)))
