import sys,ast,types,dataclasses,functools,json,hashlib
from pathlib import Path
from typing import Any,Callable,Optional
sys.dont_write_bytecode=True
O=Path(__file__).resolve().parent
import platform
W=Path('/root/dyq/workspace') if platform.machine()!='aarch64' else Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_bmm_backend_full1_r2')
sys.path[:0]=[str(W/'TileOPs'),str(W/'TileOPs/src'),'/root/dyq/workspace/.venv/lib/python3.12/site-packages','/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages','/opt/tileops-ci/runtime/lib/python3.12/site-packages']
from tileops.manifest import load_manifest,load_adts,load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate,Call
from tileops.manifest.values import ADTValue
api=O/'closure/TileOPs/benchmarks/api.py'
tree=ast.parse(api.read_bytes());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('Case','Implementation','_report_value')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(api),'exec'),globals())
def cases():
 wanted={('BmmFwdOp','square-4k-bfloat16')}
 out=[]
 for op in sorted({k[0] for k in wanted}):
  plan=entry_plan(op,load_manifest()[op],load_adts(),resolve=False)
  for row in load_workloads(op):
   for dt in row.get('dtype_cases') or [{}]:
    call=instantiate(plan,row,dt)
    if (op,call.case_id) in wanted:out.append(Case(call.case_id,call,None))
 assert len(out)==1
 return out
