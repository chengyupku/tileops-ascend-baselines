import sys,builtins,types,json
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));old=builtins.__import__
def protect(name,*a,**k):
 if name.split('.')[0] in ('torch','torch_npu','acl','tilelang','tvm'):raise AssertionError('device import '+name)
 return old(name,*a,**k)
builtins.__import__=protect
sys.addaudithook(lambda e,a: (_ for _ in ()).throw(AssertionError('dlopen')) if e=='ctypes.dlopen' else None)
import build_contracts as B
from benchmarks.ascend_all_pairs.registry import candidate_factory
from benchmarks.ascend_all_pairs.tla import PROVIDERS
from benchmarks.ascend_baselines import UnsupportedCase
cs=B.cases()
for c in cs:
 x=candidate_factory(c,'tilelang-ascend')();assert callable(x.implementation.run);x.cleanup()
 bad=types.SimpleNamespace(id=c.id,_call=c._call,params=dict(c.params,unexpected=True))
 try:candidate_factory(bad,'tilelang-ascend')()
 except UnsupportedCase:pass
 else:raise AssertionError('wrong contract accepted')
for op in ('BmmFwdOp','GemmFwdOp'):
 f=PROVIDERS[op];assert f.__module__=='_baseline700_pto23_pinned'
 assert len(f.__closure__)==1
 f.__closure__[0].cell_contents=lambda c:'previous'
 c=next(c for c in cs if c._call.signature.name==op);unknown=types.SimpleNamespace(id='unknown',_call=c._call,params=c.params)
 assert candidate_factory(unknown,'tilelang-ascend')()=='previous'
assert not any(n in sys.modules for n in ('torch','torch_npu','acl','tilelang','tvm'))
print(json.dumps(dict(passed=True,exact_routes=23,wrong_contract_rejection=True,unknown_fallback=True,lazy=True,NPU_runs=0)))
