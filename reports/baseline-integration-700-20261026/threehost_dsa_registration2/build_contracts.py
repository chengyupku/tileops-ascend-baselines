import sys,ast,types,dataclasses,functools,json,hashlib
from pathlib import Path
from typing import Any,Callable,Optional
sys.dont_write_bytecode=True
O=Path(__file__).resolve().parent
sys.path[:0]=[str(O/'manifest_closure'),'/root/dyq/workspace/.venv/lib/python3.12/site-packages']
from tileops.manifest import load_manifest,load_adts,load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate,Call
from tileops.manifest.values import ADTValue
pins=json.loads((O/'PINS.json').read_text());api=next(O/v['file'] for k,v in pins['sources'].items() if k.endswith('/benchmarks/api.py'))
tree=ast.parse(api.read_bytes());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('Case','Implementation','_report_value')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(api),'exec'),globals())
def cases():
 op='DSADecodeWithKVCacheFwdOp';wanted={'mla-d576-h128-kv96k-bfloat16','mla-d576-h128-kv128k-bfloat16'}
 plan=entry_plan(op,load_manifest()[op],load_adts(),resolve=False);out=[]
 for row in load_workloads(op):
  if row.get('label') not in ('mla-d576-h128-kv96k','mla-d576-h128-kv128k'):continue
  for dt in row.get('dtype_cases') or [{}]:
   call=instantiate(plan,row,dt)
   if call.case_id in wanted:out.append(Case(call.case_id,call,None))
 assert len(out)==2
 return out
if __name__=='__main__':
 m=types.ModuleType('registration');exec(compile((O/'registration.py').read_bytes(),str(O/'registration.py'),'exec'),m.__dict__)
 rows=[m.contract(c) for c in cases()]
 (O/'CONTRACTS.json').write_text(json.dumps(rows,indent=2)+'\n')
 hashes={n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in ['CONTRACTS.json','PINS.json','FROZEN.json']}
 s=(O/'registration.py').read_text().replace('DATA_HASHES = {}','DATA_HASHES = '+repr(hashes));(O/'registration.py').write_text(s)
 print('real contracts',[(r['case_id'],r['has_out'],r['presence'],r['rawparams']) for r in rows])
