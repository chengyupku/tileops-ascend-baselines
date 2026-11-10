import ast,builtins,pathlib,json,sys,types,dataclasses,functools,typing,importlib.util,copy
T=pathlib.Path(__file__).resolve().parent;W=pathlib.Path('/root/dyq/workspace');sys.dont_write_bytecode=True
sys.path[:0]=[str(W/'TileOPs'),str(W/'TileOPs/src'),str(W/'.venv/lib/python3.12/site-packages')]
original=builtins.__import__;blocked=[]
def checked(name,*a,**k):
 if name.split('.')[0] in ('torch','torch_npu','acl'):blocked.append(name);raise AssertionError('Device import during CPU routing '+name)
 return original(name,*a,**k)
builtins.__import__=checked
import ctypes
ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(AssertionError('native load'))
spec=importlib.util.spec_from_file_location('moe_up_private',T/'integration700_moe_up.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from tileops.manifest import load_manifest,load_adts,load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate
from tileops.manifest.values import ADTValue
plan=entry_plan('MoEGroupedGemmFwdOp',load_manifest()['MoEGroupedGemmFwdOp'],load_adts(),resolve=False)
calls={c.case_id:c for r in load_workloads('MoEGroupedGemmFwdOp') for d in r.get('dtype_cases') or [{}] if (c:=instantiate(plan,r,d)).case_id in {k[2] for k in m.CASE_PARAMS}}
nodes=[n for n in ast.parse((W/'TileOPs/benchmarks/api.py').read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('Case','_report_value')]
ns=dict(dataclasses=dataclasses,functools=functools,Any=typing.Any,Optional=typing.Optional,Callable=typing.Callable,Call=type(next(iter(calls.values()))),ADTValue=ADTValue)
exec(compile(ast.Module(body=nodes,type_ignores=[]),'original-Case','exec'),ns)
from benchmarks.ascend_all_pairs import ops_transformer
from benchmarks.ascend_all_pairs.registry import candidate_factory
from benchmarks.ascend_baselines import UnsupportedCase
prior=ops_transformer.PROVIDERS['MoEGroupedGemmFwdOp'];ops_transformer.PROVIDERS['MoEGroupedGemmFwdOp']=m.wrap(prior)
results=[]
for c in calls.values():
 case=ns['Case'](c.case_id,c,None);assert m.matches(case)
 candidate=candidate_factory(case,'ops-transformer')();assert candidate.api=='ops-transformer::aclnnGroupedMatmulV5';assert callable(candidate.cleanup)
 try:candidate.implementation.run()
 except RuntimeError as e:assert 'guard' in str(e)
 else:raise AssertionError('unguarded run allowed')
 candidate.cleanup()
 assert not m.matches(case,'ops-nn')
 bad=types.SimpleNamespace(id=case.id,params=json.loads(json.dumps(case.params)),_call=case._call);bad.params['a'][0][0]+=1
 assert not m.matches(bad)
 try:ops_transformer.PROVIDERS['MoEGroupedGemmFwdOp'](bad)
 except UnsupportedCase:pass
 else:raise AssertionError('invalid known case fell through')
 unknown=types.SimpleNamespace(id='not-owned',_call=case._call)
 sentinel=object();assert m.wrap(lambda _:sentinel)(unknown) is sentinel
 results.append({'key':['ops-transformer','MoEGroupedGemmFwdOp',case.id],'real_manifest_call':True,'real_registry_factory':True,'cpu_lazy':True,'unguarded_run_rejected':True,'changed_shape_rejected':True,'other_baseline_rejected':True,'unknown_key_delegated':True})
ops_transformer.PROVIDERS['MoEGroupedGemmFwdOp']=prior
assert not blocked
(T/'REGISTRATION_CPU_CHECKS.json').write_text(json.dumps({'passed':True,'results':results,'device_imports':blocked,'npu_runs':0,'native_loads':0,'global_files_changed':False},indent=2));print('real registry CPU route checks passed',len(results))
