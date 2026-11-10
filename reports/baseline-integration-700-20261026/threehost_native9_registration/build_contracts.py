import sys,ast,types,dataclasses,functools,json,hashlib
from pathlib import Path
from typing import Any,Callable,Optional
sys.dont_write_bytecode=True
O=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/threehost_native9_registration')
sys.path[:0]=['/root/dyq/workspace/TileOPs','/root/dyq/workspace/TileOPs/src','/root/dyq/workspace/.venv/lib/python3.12/site-packages']
from tileops.manifest import load_manifest,load_adts,load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate,Call
from tileops.manifest.values import ADTValue
api=O/'closure/TileOPs/benchmarks/api.py'
tree=ast.parse(api.read_bytes());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('Case','Implementation','_report_value')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(api),'exec'),globals())
def cases():
 wanted={(r['op'],r['case_id']) for r in json.loads((O.parent/'threehost_native_completion9/CONTRACTS.json').read_text())}
 out=[]
 for op in sorted({k[0] for k in wanted}):
  plan=entry_plan(op,load_manifest()[op],load_adts(),resolve=False)
  for row in load_workloads(op):
   for dt in row.get('dtype_cases') or [{}]:
    call=instantiate(plan,row,dt)
    if (op,call.case_id) in wanted:out.append(Case(call.case_id,call,None))
 assert len(out)==9
 return out
if __name__=='__main__':
 m=types.ModuleType('registration');exec(compile((O/'registration.py').read_bytes(),str(O/'registration.py'),'exec'),m.__dict__)
 rows=[m.contract(c) for c in cases()]
 (O/'CONTRACTS.json').write_text(json.dumps(rows,indent=2)+'\n')
 pins={n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in ['CONTRACTS.json','PINS.json','FROZEN_fp8.json','FROZEN_bf16arm2.json']}
 src=(O/'registration.py').read_text();src=src.replace('DATA_HASHES = {}  # Filled by build_contracts.py after freezing data.','DATA_HASHES = '+repr(pins))
 (O/'registration.py').write_text(src)
 print('real contracts',len(rows))
