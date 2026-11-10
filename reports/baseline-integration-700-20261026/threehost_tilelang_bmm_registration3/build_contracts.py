import sys,ast,types,dataclasses,functools,json,hashlib
from pathlib import Path
from typing import Any,Callable,Optional
sys.dont_write_bytecode=True
O=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/threehost_tilelang_bmm_registration3')
sys.path[:0]=['/root/dyq/workspace/TileOPs','/root/dyq/workspace/TileOPs/src','/root/dyq/workspace/.venv/lib/python3.12/site-packages']
from tileops.manifest import load_manifest,load_adts,load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate,Call
from tileops.manifest.values import ADTValue
api=O/'closure/99d43801a34566e59c1ae519adb53705e9e228cd14f1eb950219698a393f1772'
tree=ast.parse(api.read_bytes());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('Case','Implementation','_report_value')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(api),'exec'),globals())
def cases():
 wanted={(r['operator'],r['case_id']) for r in json.loads((O/'SCOPE.json').read_text())}
 out=[]
 for op in sorted({k[0] for k in wanted}):
  plan=entry_plan(op,load_manifest()[op],load_adts(),resolve=False)
  for row in load_workloads(op):
   for dt in row.get('dtype_cases') or [{}]:
    call=instantiate(plan,row,dt)
    if (op,call.case_id) in wanted:out.append(Case(call.case_id,call,None))
 assert len(out)==3
 return out

if __name__=="__main__":
 import registration
 (O/"CONTRACTS.json").write_text(json.dumps([registration.contract(c) for c in cases()],indent=2)+"\n")
