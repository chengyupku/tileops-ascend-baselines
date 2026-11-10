import ast, hashlib, json, sys, dataclasses, functools
from pathlib import Path
P=Path(__file__).resolve().parent
OLD=P.parent/'threehost_dense48_r2'
REMOTE='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_dense48_r2'
def dump(n,x): (P/n).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
sp=json.loads((P/'SOURCE_PINS.json').read_text())
fc=json.loads((OLD/'FROZEN_CHECKER.json').read_text())
for r in fc:
 n=r['path'].split(REMOTE+'/')[-1]
 if n in sp: assert sp[n]==r['sha256']
assert sp['binding.py']==hashlib.sha256((OLD/'binding.py').read_bytes()).hexdigest()
x=json.loads((OLD/'checker.json').read_text())
vendor={REMOTE+'/'+n:h for n,h in x['source_sha256'].items() if n.startswith('vendor/') and '/ascendc/' not in n}
loaded={r['path']:r['sha256'] for r in json.loads((OLD/'LOADED_DEPENDENCIES.json').read_text()) if '/Ascend/' in r['path'] or '/torch/' in r['path'] or '/torch_npu/' in r['path'] or '/vendor/' in r['path']}
dump('RUNTIME_PINS.json',dict(vendor=vendor,loaded=loaded,source={REMOTE+'/'+n:h for n,h in sp.items()}))
sys.path[:0]=[str(Path('/root/dyq/workspace/TileOPs')),str(Path('/root/dyq/workspace/TileOPs/src'))]
from tileops.manifest import load_manifest,load_adts,load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate,Call
from tileops.manifest.values import ADTValue
plan=entry_plan('GemmFP8FwdOp',load_manifest()['GemmFP8FwdOp'],load_adts(),resolve=False)
c=next(c for r in load_workloads('GemmFP8FwdOp') for d in r.get('dtype_cases') or [{}] if (c:=instantiate(plan,r,d)).case_id==x['case_id'])
from typing import Any,Callable,Optional
tree=ast.parse((P/'frozen/api.py').read_text());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('Case','Implementation','_report_value')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'frozen/api.py','exec'))
case=Case(c.case_id,c,None)
contract=dict(key=['ops-transformer','GemmFP8FwdOp',c.case_id],params=case.params,call=dict(indices=c.indices,rawparams=c.params,tensors=c.tensors,specs={n:None if s is None else dataclasses.asdict(s) for n,s in c.specs.items()},presence={n:c.present(n) for n in set(c.specs)|{'out'}},has_out=c.has_out,signature=dataclasses.asdict(c.signature)))
assert json.loads(json.dumps(case.params))==x['case_snapshot']['parameters']
dump('EXACT_CONTRACT.json',contract)
dump('LOCAL_PINS.json',{str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list((P/'frozen').glob('*.py'))+[P/'EXACT_CONTRACT.json',P/'RUNTIME_PINS.json',P/'SOURCE_PINS.json']})
print('vendor',len(vendor),'loaded runtime',len(loaded),'exact contract verified against accepted checker')
