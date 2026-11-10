import builtins,hashlib,json,os,sys,types,mmap,tempfile
from pathlib import Path
O=Path(__file__).resolve().parent
sys.dont_write_bytecode=True

sys.path[:0]=['/root/dyq/workspace/TileOPs','/root/dyq/workspace/TileOPs/src','/root/dyq/workspace/.venv/lib/python3.12/site-packages']
original_import=builtins.__import__; forbidden=[]
def protected(name,*a,**kw):
 if name.split('.')[0] in ('torch','torch_npu','acl'):
  forbidden.append(name);raise AssertionError('Forbidden device import '+name)
 return original_import(name,*a,**kw)
builtins.__import__=protected
import ctypes
native_loads=[]
def no_native(*a,**kw):
 native_loads.append(str(a));raise AssertionError('Native library load forbidden')
ctypes.CDLL=no_native
ctypes.PyDLL=no_native
M=types.ModuleType('registration');M.__file__=str(O/'integration700_bf16_moe_down_registration3.py')
exec(compile(Path(M.__file__).read_bytes(),M.__file__,'exec'),M.__dict__)
checks=[]
def passed(x):
 checks.append(x)
 import datetime
 (O/'HEARTBEAT.json').write_text(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=x,npu_runs=0)))
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
op='MoEGroupedGemmFwdOp'
plan=entry_plan(op,load_manifest()[op],load_adts(),resolve=False)
realcalls={c.case_id:c for r in load_workloads(op) for d in r.get('dtype_cases') or [{}] if (c:=instantiate(plan,r,d)).case_id in {k[2] for k in M.CASE_PARAMS}}
assert len(realcalls)==3
realcall=realcalls['ds-v3-decode-down-bfloat16']
api_source=Path('/root/dyq/workspace/TileOPs/benchmarks/api.py').read_bytes()
tree=ast.parse(api_source)
nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('Case','_report_value')]
ns=dict(dataclasses=dataclasses,functools=functools,Any=typing.Any,Optional=typing.Optional,Callable=typing.Callable,Call=type(realcall),ADTValue=ADTValue)
exec(compile(ast.Module(body=nodes,type_ignores=[]),'original-api-CPU-classes','exec'),ns)
cs=[ns['Case'](c.case_id,c,None) for c in realcalls.values()]
realcase=next(c for c in cs if c.id==realcall.case_id)
from benchmarks.ascend_baselines import UnsupportedCase

def public_reject(fn):
 try:fn()
 except UnsupportedCase as e:assert type(e) is UnsupportedCase;return
 raise AssertionError('Expected canonical public UnsupportedCase')

def case(k,p):
 raw=realcalls.get(k[2],realcall)
 return types.SimpleNamespace(id=k[2],params=p,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=k[1]),indices=dict(raw.indices),params=dict(raw.params),tensors=dict(raw.tensors),present=raw.present))

from benchmarks.ascend_all_pairs.registry import candidate_factory
import importlib
published=importlib.import_module('benchmarks.ascend_all_pairs.ops_transformer')
for c in cs:
 assert callable(candidate_factory(c,'ops-transformer')().implementation.run)
 bad=case(('ops-transformer',op,c.id),dict(c.params,unexpected=True))
 public_reject(lambda:candidate_factory(bad,'ops-transformer')())
dispatch=published.PROVIDERS[op];previous=dispatch.previous
assert previous is not None and dispatch.__module__.endswith('.integration700_bf16_moe_down_registration3')
for cell in dispatch.__closure__:
 if cell.cell_contents is previous:cell.cell_contents=lambda c:('previous',c.id)
unknown=types.SimpleNamespace(id='unknown-key',params={},_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)))
assert candidate_factory(unknown,'ops-transformer')()==('previous','unknown-key')
assert not forbidden and not native_loads
print(json.dumps(dict(passed=True,published_exact_routes=len(cs),real_manifest_cases=True,public_exception_type=True,wrong_params_rejected=len(cs),previous_routes_preserved=True,device_imports=0,numerical_runs=0)))
