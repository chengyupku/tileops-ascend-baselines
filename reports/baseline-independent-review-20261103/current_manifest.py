import sys,types,pathlib,json,hashlib
sys.dont_write_bytecode=True
R=pathlib.Path('/root/dyq/workspace');P=R/'TileOPs/src/tileops';O=pathlib.Path(__file__).parent
p=types.ModuleType('tileops');p.__path__=[str(P)];sys.modules['tileops']=p
from tileops.manifest import load_manifest,load_adts,load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate
from tileops.manifest.values import ADTValue
out=[]; errors=[]
for name,entry in load_manifest().items():
 try:
  plan=entry_plan(name,entry,load_adts(),resolve=False)
  for row in load_workloads(name):
   for dt in row.get('dtype_cases') or [{}]:
    call=instantiate(plan,row,dt)
    out.append({'op':name,'case_id':call.case_id,'family':entry.get('family'),'tensors':call.tensors,'params':call.tensors|{k:({v.kind:dict(v.fields)} if isinstance(v,ADTValue) else v) for k,v in call.params.items()}})
 except Exception as e: errors.append([name,repr(e)])
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
(O/'CURRENT_MANIFEST.json').write_text(json.dumps({'cases':out,'errors':errors,'sources':{str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in P.joinpath('manifest').rglob('*') if f.is_file() and f.suffix in ('.py','.yaml')}},default=str))
print(len(out),errors)
