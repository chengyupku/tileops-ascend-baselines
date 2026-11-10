import ast,sys,types,dataclasses,functools,json
from pathlib import Path
from typing import Any,Callable,Optional
sys.dont_write_bytecode=True
O=Path(__file__).resolve().parent;W=O.parents[2]
sys.path[:0]=[str(W/'TileOPs'),str(W/'TileOPs/src'),str(W/'.venv/lib/python3.12/site-packages')]
sys.path.extend([str(W/'.cache/all-pairs/remote_tla_compiler/python'),str(W/'.cache/all-pairs/remote_native_tla/python-deps'),str(W/'.cache/all-pairs/remote_triton_runtime/overlay/site-packages'),'/opt/tileops-ci/runtime/lib/python3.12/site-packages'])
from tileops.manifest import load_manifest,load_adts,load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate,Call
from tileops.manifest.values import ADTValue
api=O/'closure/api.py'
nodes=[n for n in ast.parse(api.read_bytes()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('Case','Implementation','_report_value')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(api),'exec'),globals())
def cases():
 wanted={tuple(r['key'][1:]) for r in json.loads((O/'EVIDENCE.json').read_text())};out=[]
 for op in ('BmmFwdOp','GemmFwdOp'):
  plan=entry_plan(op,load_manifest()[op],load_adts(),resolve=False)
  for row in load_workloads(op):
   for dt in row.get('dtype_cases') or [{}]:
    call=instantiate(plan,row,dt)
    if (op,call.case_id) in wanted:out.append(Case(call.case_id,call,None))
 assert len(out)==23
 return out
if __name__=='__main__':
 import registration
 (O/'CONTRACTS.json').write_text(json.dumps([registration.contract(c) for c in cases()],indent=2)+'\n')
