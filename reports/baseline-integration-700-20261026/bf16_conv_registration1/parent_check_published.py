import builtins,hashlib,json,os,sys,types,mmap,tempfile
from pathlib import Path
O=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_conv_registration1')
sys.path[:0]=['/root/dyq/workspace/TileOPs','/root/dyq/workspace/TileOPs/src','/root/dyq/workspace/.venv/lib/python3.12/site-packages']
original_import=builtins.__import__; forbidden=[]
def protected(name,*a,**kw):
 if name.split('.')[0] in ('torch','torch_npu','acl'):
  forbidden.append(name);raise AssertionError('Forbidden device import '+name)
 return original_import(name,*a,**kw)
builtins.__import__=protected
M=types.ModuleType('registration');M.__file__=str(O/'integration700_bf16_conv_registration1.py')
exec(compile(Path(M.__file__).read_bytes(),M.__file__,'exec'),M.__dict__)
checks=[]
def passed(x):checks.append(x)
def reject(fn):
 try:fn()
 except (RuntimeError,ValueError,AssertionError,KeyError):return
 raise AssertionError('Expected rejection')
# Instantiate the actual manifest Call, and the unmodified Case class AST.
# This excludes only api module imports which would import torch at module scope.
import ast,dataclasses,functools,typing,copy
from tileops.manifest import load_manifest,load_adts,load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate
from tileops.manifest.values import ADTValue
op='Conv1dFwdOp'
plan=entry_plan(op,load_manifest()[op],load_adts(),resolve=False)
realcall=next(c for c in (instantiate(plan,r,d) for r in load_workloads(op) for d in r.get('dtype_cases') or [{}]) if c.case_id=='whisper-conv1-bias-bfloat16')
api_source=Path('/root/dyq/workspace/TileOPs/benchmarks/api.py').read_bytes()
tree=ast.parse(api_source)
nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('Case','_report_value')]
ns=dict(dataclasses=dataclasses,functools=functools,Any=typing.Any,Optional=typing.Optional,Callable=typing.Callable,Call=type(realcall),ADTValue=ADTValue)
exec(compile(ast.Module(body=nodes,type_ignores=[]),'original-api-CPU-classes','exec'),ns)
realcase=ns['Case'](realcall.case_id,realcall,None)
assert isinstance(realcase.params['input'][0],tuple)
def case(k,p):
 return types.SimpleNamespace(id=k[2],params=p,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=k[1]),indices=dict(realcall.indices),params=dict(realcall.params),tensors=dict(realcall.tensors),present=realcall.present))
cs=[realcase]
for c in cs:
 assert M.matches(c); candidate=M.make(c); assert callable(candidate.implementation.run)
 assert not M.matches(c,'wrong-baseline')
 bad=case(('ops-transformer','Conv1dFwdOp',c.id),dict(c.params,unexpected=True));assert not M.matches(bad);reject(lambda:M.make(bad))

from benchmarks.ascend_all_pairs.registry import candidate_factory
import importlib
published=importlib.import_module('benchmarks.ascend_all_pairs.ops_transformer')
from benchmarks.ascend_baselines import UnsupportedCase
for c in cs:
 assert callable(candidate_factory(c,'ops-transformer')().implementation.run)
 bad=types.SimpleNamespace(id=c.id,params=dict(c.params,unexpected=True),_call=c._call)
 try:candidate_factory(bad,'ops-transformer')()
 except UnsupportedCase:pass
 else:raise AssertionError('wrong parameters accepted')
for op in ('Conv1dFwdOp',):
 dispatch=published.PROVIDERS[op];previous=dispatch.previous;assert previous is not None
 assert dispatch.__module__.endswith('.integration700_bf16_conv_registration1')
 for cell in dispatch.__closure__:
  if cell.cell_contents is previous:cell.cell_contents=lambda c:('previous',c.id)
 c=types.SimpleNamespace(id='unknown-key',params={},_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)))
 assert candidate_factory(c,'ops-transformer')()==('previous','unknown-key')
assert not forbidden
print(json.dumps(dict(passed=True,published_exact_routes=1,real_manifest_cases=True,wrong_params_rejected=1,previous_routes_preserved=1,device_imports=0,numerical_runs=0)))
