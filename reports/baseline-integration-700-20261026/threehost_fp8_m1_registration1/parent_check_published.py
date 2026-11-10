"""CPU only: real manifest/Case, hostile imports, new route/lifetime behavior."""
import ast,builtins,copy,dataclasses,functools,hashlib,json,sys,tempfile,types,shutil
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(P),'/root/dyq/workspace/TileOPs','/root/dyq/workspace/TileOPs/src']
original=builtins.__import__
def guarded_import(name,*args,**kwargs):
    if name.split('.')[0] in ('torch','torch_npu','acl'):
        raise AssertionError('Forbidden CPU import: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded_import
from tileops.manifest import load_manifest,load_adts,load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate,Call
from tileops.manifest.values import ADTValue
from typing import Any,Callable,Optional
from benchmarks.ascend_baselines import UnsupportedCase
import benchmarks.ascend_all_pairs.ops_transformer._integration700_fp8_m1 as m
nodes=[n for n in ast.parse((P/'frozen/api.py').read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('Case','_report_value')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'original Case AST','exec'))
plan=entry_plan(m.KEY[1],load_manifest()[m.KEY[1]],load_adts(),resolve=False)
call=next(c for r in load_workloads(m.KEY[1]) for d in r.get('dtype_cases') or [{}] if (c:=instantiate(plan,r,d)).case_id==m.KEY[2])
case=Case(call.case_id,call,None)

import sys
sys.addaudithook(lambda event,args: (_ for _ in ()).throw(AssertionError('Native load forbidden')) if event=='ctypes.dlopen' else None)
from benchmarks.ascend_all_pairs.registry import candidate_factory
import benchmarks.ascend_all_pairs.ops_transformer as provider
candidate=candidate_factory(case,'ops-transformer')(); assert callable(candidate.implementation.run);candidate.cleanup()
bad=copy.deepcopy(call);bad.indices['M']=2
try:candidate_factory(Case(bad.case_id,bad,None),'ops-transformer')()
except UnsupportedCase as ex:assert type(ex) is UnsupportedCase
else:raise AssertionError('Wrong exactkey parameters accepted')
route=provider.PROVIDERS['GemmFP8FwdOp'];assert route.__module__==m.__name__;assert route.previous is not None
previous=route.previous
for cell in route.__closure__:
 if cell.cell_contents is previous:cell.cell_contents=lambda c:'previous-preserved'
unknown=Case('unknown',call,None);assert candidate_factory(unknown,'ops-transformer')()=='previous-preserved'
assert not any(n in sys.modules for n in ('torch','torch_npu','acl'))
print(json.dumps(dict(passed=True,published_exact_routes=1,real_manifest=True,public_exception_type=True,previous_preserved=True,CPU_lazy=True,NPU_runs=0)))
