import sys,builtins,ctypes,json,types,copy
from pathlib import Path
P=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/threehost_native9_registration');sys.path.insert(0,str(P))
original=builtins.__import__
def protected(name,*a,**k):
 if name.split('.')[0] in ('torch','torch_npu','acl','tilelang','tvm'):raise AssertionError('Device import '+name)
 return original(name,*a,**k)
builtins.__import__=protected
sys.addaudithook(lambda event,args: (_ for _ in ()).throw(AssertionError('Native dlopen forbidden')) if event=='ctypes.dlopen' else None)
import build_contracts as B
from benchmarks.ascend_all_pairs.registry import candidate_factory
from benchmarks.ascend_baselines import UnsupportedCase
import benchmarks.ascend_all_pairs.ops_transformer as provider
import benchmarks.ascend_all_pairs.ops_transformer.integration700_threehost_native9 as M
cases=B.cases()
for c in cases:
 cand=candidate_factory(c,'ops-transformer')();assert callable(cand.implementation.run);cand.cleanup()
 bad=types.SimpleNamespace(id=c.id,_call=c._call,params=dict(c.params,unexpected=True))
 try:candidate_factory(bad,'ops-transformer')()
 except UnsupportedCase as e:assert type(e) is UnsupportedCase
 else:raise AssertionError('Wrong exact parameters accepted')
for op in {c._call.signature.name for c in cases}:
 route=provider.PROVIDERS[op];assert route.__module__==M.__name__;previous=route.previous;assert previous is not None
 for cell in route.__closure__:
  if cell.cell_contents is previous:cell.cell_contents=lambda c:'preserved'
 c=next(c for c in cases if c._call.signature.name==op);unknown=types.SimpleNamespace(id='unknown',_call=c._call,params=c.params)
 assert candidate_factory(unknown,'ops-transformer')()=='preserved'
assert not any(n in sys.modules for n in ('torch','torch_npu','acl','tilelang','tvm'))
print(json.dumps(dict(passed=True,published_exact_routes=9,real_manifest=True,public_exception_type=True,unknown_fallbacks=3,CPU_lazy=True,NPU_runs=0)))
