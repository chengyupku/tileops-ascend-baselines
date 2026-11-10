import builtins,hashlib,json,os,sys,types,mmap,tempfile
from pathlib import Path
O=Path(__file__).resolve().parent
sys.path[:0]=['/root/dyq/workspace/TileOPs','/root/dyq/workspace/.venv/lib/python3.12/site-packages']
original_import=builtins.__import__; forbidden=[]; native_load_attempts=[]
import ctypes
def audit(event,args):
 if event == 'ctypes.dlopen':
  native_load_attempts.append(str(args));raise AssertionError('Native library load forbidden in CPU checks')
sys.addaudithook(audit)
def protected(name,*a,**kw):
 if name.split('.')[0] in ('torch','torch_npu','acl'):
  forbidden.append(name);raise AssertionError('Forbidden device import '+name)
 return original_import(name,*a,**kw)
builtins.__import__=protected
M=types.ModuleType('registration');M.__file__=str(O/'integration700_fp16_grouped_registration2.py')
exec(compile(Path(M.__file__).read_bytes(),M.__file__,'exec'),M.__dict__)
checks=[]
def passed(x):checks.append(x)
def reject(fn):
 try:fn()
 except (RuntimeError,ValueError,AssertionError,KeyError):return
 raise AssertionError('Expected rejection')
# Instantiate original CPU-only manifest Calls, then execute exactly the passive
# Case/Implementation declarations from original API (its module imports torch).
import ast, dataclasses, functools
from typing import Any, Callable, Optional
sys.path.insert(0,'/root/dyq/workspace/TileOPs/src')
from tileops.manifest import load_manifest,load_adts,load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate,Call
from tileops.manifest.values import ADTValue
api_path=Path('/root/dyq/workspace/TileOPs/benchmarks/api.py')
tree=ast.parse(api_path.read_bytes()); nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('Case','Implementation','_report_value')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(api_path),'exec'),globals())
cs=[]
for op in ('GroupedGemmFwdOp',):
 plan=entry_plan(op,load_manifest()[op],load_adts(),resolve=False)
 for row in load_workloads(op):
  for dtype in row.get('dtype_cases') or [{}]:
   call=instantiate(plan,row,dtype)
   if ('ops-transformer',op,call.case_id) in M.CASE_PARAMS:cs.append(Case(call.case_id,call,None))
assert len(cs)==2
passed('two real manifest Call instances and exact original passive Case declarations; no workload or inputs accessed')
def case(k,p):
 c=next(c for c in cs if c.id==k[2] or 'old-'+c.id==k[2])
 return types.SimpleNamespace(id=k[2],params=p,_call=c._call)
for c in cs:
 assert M.matches(c); candidate=M.make(c); assert callable(candidate.implementation.run)
 assert not M.matches(c,'wrong-baseline')
 bad=case(('ops-transformer',c._call.signature.name,c.id),dict(c.params,unexpected=True));assert not M.matches(bad);reject(lambda:M.make(bad))
passed('2 real canonical params CPU construction; wrong baseline/full params rejected')

from benchmarks.ascend_all_pairs.registry import candidate_factory
from benchmarks.ascend_baselines import UnsupportedCase
import importlib
published=importlib.import_module('benchmarks.ascend_all_pairs.ops_transformer')
for c in cs:
 assert callable(candidate_factory(c,'ops-transformer')().implementation.run)
 bad=case(('ops-transformer','GroupedGemmFwdOp',c.id),dict(c.params,unexpected=True))
 try:candidate_factory(bad,'ops-transformer')()
 except UnsupportedCase as ex:assert type(ex) is UnsupportedCase
 else:raise AssertionError('Wrong params accepted')
dispatch=published.PROVIDERS['GroupedGemmFwdOp'];previous=dispatch.previous
assert previous is not None and dispatch.__module__.endswith('.integration700_fp16_grouped_registration2')
for cell in dispatch.__closure__:
 if cell.cell_contents is previous:cell.cell_contents=lambda c:('previous',c.id)
unknown=types.SimpleNamespace(id='unknown-key',params={},_call=types.SimpleNamespace(signature=types.SimpleNamespace(name='GroupedGemmFwdOp')))
assert candidate_factory(unknown,'ops-transformer')()==('previous','unknown-key')
assert not forbidden and not native_load_attempts
print(json.dumps(dict(passed=True,published_exact_routes=len(cs),real_manifest_cases=True,public_exception_type=True,wrong_params_rejected=len(cs),previous_routes_preserved=True,device_imports=0,numerical_runs=0)))
