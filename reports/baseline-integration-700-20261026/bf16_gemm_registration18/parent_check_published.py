import builtins,sys,json,types,importlib
sys.dont_write_bytecode=True
sys.path.insert(0,"/root/dyq/workspace/TileOPs")
old_import=builtins.__import__
def protect(n,*a,**kw):
 if n.split('.')[0] in ('torch','torch_npu','acl'):raise AssertionError('device import '+n)
 return old_import(n,*a,**kw)
builtins.__import__=protect
from benchmarks.ascend_all_pairs.registry import candidate_factory
p=importlib.import_module('benchmarks.ascend_all_pairs.ops_transformer')
m=importlib.import_module('benchmarks.ascend_all_pairs.ops_transformer.integration700_bf16_gemm_registration18')
from benchmarks.ascend_baselines import UnsupportedCase
for k,v in m.CASE_PARAMS.items():
 c=types.SimpleNamespace(id=k[2],params=json.loads(v),_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=k[1])))
 out=candidate_factory(c,k[0])();assert callable(out.implementation.run)
 c.params['unexpected']=True
 try:candidate_factory(c,k[0])()
 except UnsupportedCase:pass
 else:raise AssertionError('wrong params accepted')
assert p.PROVIDERS['GemmFwdOp'].previous is not None
# Actual published wrapper delegation, with the retained previous callable replaced only in this process.
prev=p.PROVIDERS['GemmFwdOp'].previous
assert any(cell.cell_contents is prev for cell in p.PROVIDERS['GemmFwdOp'].__closure__)
for cell in p.PROVIDERS['GemmFwdOp'].__closure__:
 if cell.cell_contents is prev:cell.cell_contents=lambda c:('previous',c.id)
c=types.SimpleNamespace(id='unlisted-exact-key',params={},_call=types.SimpleNamespace(signature=types.SimpleNamespace(name='GemmFwdOp')))
assert candidate_factory(c,'ops-transformer')()==('previous',c.id)
assert 'torch_npu' not in sys.modules
print(json.dumps(dict(passed=True,actual_registry_exact_routes=18,wrong_params_rejected=18,previous_callable_retained=True,device_imports=0,numerical_runs=0)))
