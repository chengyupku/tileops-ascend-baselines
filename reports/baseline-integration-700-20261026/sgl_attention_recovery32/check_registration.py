from pathlib import Path
import sys,json,types,importlib.abc
P=Path(__file__).resolve().parent;W=P.parents[2];sys.path[:0]=[str(P),str(W/'TileOPs')];sys.dont_write_bytecode=True
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','triton'):raise RuntimeError('CPU-only device import blocked '+fullname)
sys.meta_path.insert(0,Block())
from benchmarks.ascend_all_pairs import registry
sgl=types.ModuleType('benchmarks.ascend_all_pairs.sgl');sys.modules[sgl.__name__]=sgl
rows=json.loads((P/'REGISTRATION_SCOPE.json').read_text())['rows'];ops={r['operator'] for r in rows};sentinel=object();sgl.PROVIDERS={op:(lambda case:sentinel) for op in ops};exec(compile((P/'provider_registration.append.py').read_text(),str(P/'provider_registration.append.py'),'exec'),sgl.__dict__)
class Reached(Exception):pass
seen=[]
def trace(frame,event,arg):
 if event=='call' and Path(frame.f_code.co_filename).parent==P and frame.f_code.co_name in ('build','write','varlen','mla_decode','mha_paged','prefill_append'):
  seen.append({'file':frame.f_code.co_filename,'function':frame.f_code.co_name});raise Reached()
 return trace
checked=[]
for r in rows:
 if r['status']!='verified_unregistered':continue
 case=types.SimpleNamespace(id=r['case_id'],_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=r['operator'])))
 sys.settrace(trace)
 try:registry.candidate_factory(case,r['baseline'])()
 except Reached:pass
 else:raise AssertionError('Actual adapter not reached')
 finally:sys.settrace(None)
 checked.append({'key':r['key'],'entry':seen[-1]})
for op in ops:
 case=types.SimpleNamespace(id='__outside_exact_scope__',_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)))
 assert registry.candidate_factory(case,'sgl-kernel-npu')() is sentinel
(P/'REGISTRATION_CHECK.json').write_text(json.dumps({'cpu_only':True,'actual_registry':registry.__file__,'verified_exact_keys':checked,'fallbacks_preserved':len(ops),'numerical_reexecution':False},indent=2));print('checked',len(checked),'exact routes and',len(ops),'fallbacks')
