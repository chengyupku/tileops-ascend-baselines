import sys,ast,types,dataclasses,functools,json,hashlib
from pathlib import Path
from typing import Any,Callable,Optional
sys.dont_write_bytecode=True
O=Path(__file__).resolve().parent
sys.path[:0]=['/root/dyq/workspace/TileOPs','/root/dyq/workspace/TileOPs/src','/root/dyq/workspace/.venv/lib/python3.12/site-packages']
from tileops.manifest import load_manifest,load_adts,load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate,Call
from tileops.manifest.values import ADTValue
api=O/'closure/api.py'
tree=ast.parse(api.read_bytes());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('Case','Implementation','_report_value')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(api),'exec'),globals())
def cases():
 wanted={r['case_id'] for r in json.loads((O/'SCOPE.json').read_text())}
 out=[];op='DeltaNetFwdOp';plan=entry_plan(op,load_manifest()[op],load_adts(),resolve=False)
 for row in load_workloads(op):
  for dt in row.get('dtype_cases') or [{}]:
   call=instantiate(plan,row,dt)
   if call.case_id in wanted:out.append(Case(call.case_id,call,None))
 assert len(out)==8
 return out
if __name__=='__main__':
 import registration as m
 rows=[m.contract(c) for c in cases()]
 (O/'CONTRACTS.json').write_text(json.dumps(rows,indent=2)+'\n')
 pins={n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in ['CONTRACTS.json','LOCAL_PINS.json','RUNTIME_PINS.json']}
 src=(O/'registration.py').read_text();src=src.replace('DATA_HASHES = {}', 'DATA_HASHES = '+repr(pins));(O/'registration.py').write_text(src)
 print('real contracts',len(rows))
