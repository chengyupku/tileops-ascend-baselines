from pathlib import Path
O=Path(__file__).resolve().parent
s=(O.parent/'bf16_gemm_registration18/cpu_checks.py').read_text().replace('integration700_bf16_gemm_registration18','integration700_bf16_bmm_registration7').replace('BF16_GEMM','BF16_BMM')
a=s.index('def case(');b=s.index('for c in cs:')
s=s[:a]+'''# Instantiate original CPU-only manifest Calls, then execute exactly the passive
# Case/Implementation declarations from original API (its module imports torch).
import ast, dataclasses, functools
from typing import Any, Callable, Optional
sys.path.insert(0,'/root/dyq/workspace/TileOPs/src')
from tileops.manifest import load_manifest,load_adts,load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate,Call
from tileops.manifest.values import ADTValue
api_path=Path('/root/dyq/workspace/TileOPs/benchmarks/api.py')
tree=ast.parse(api_path.read_bytes()); nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('Case','Implementation','_report_value')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(api_path),'exec'),globals())
cs=[]
for op in ('BmmFwdOp','GroupedGemmFwdOp'):
 plan=entry_plan(op,load_manifest()[op],load_adts(),resolve=False)
 for row in load_workloads(op):
  for dtype in row.get('dtype_cases') or [{}]:
   call=instantiate(plan,row,dtype)
   if ('ops-transformer',op,call.case_id) in M.CASE_PARAMS:cs.append(Case(call.case_id,call,None))
assert len(cs)==7
passed('seven real manifest Call instances and exact original passive Case declarations; no workload or inputs accessed')
def case(k,p):
 c=next(c for c in cs if c._call.signature.name==k[1])
 return types.SimpleNamespace(id=k[2],params=p,_call=c._call)
''' + s[b:]
s=s.replace("bad=case(('ops-transformer','GemmFwdOp',c.id),dict(c.params,unexpected=True))","bad=case(('ops-transformer',c._call.signature.name,c.id),dict(c.params,unexpected=True))")
s=s.replace("unknown=case(('ops-transformer','GemmFwdOp','old-'+c.id),c.params)","unknown=case(('ops-transformer',c._call.signature.name,'old-'+c.id),c.params)")
s=s.replace('==18','==7').replace('18 real canonical','7 real canonical').replace('18 previous-route','7 previous-route')
s=s.replace("'benchmarks.ascend_baselines')}","'benchmarks.ascend_baselines','adapter','benchmarks.api')}")
s=s.replace("({'binding':types.SimpleNamespace(dense_fp32=build)},run_dir,identity,environment)","({'binding':types.SimpleNamespace(dense_fp32=build),'adapter':types.SimpleNamespace(candidate=adapt)},run_dir,identity,environment)")
s=s.replace('def build(c):', '''def adapt(c,*,native_factory):
 assert dict(c._call.indices)==dict(cs[0]._call.indices)
 return native_factory(c)
def build(c):''')
s=s.replace("passed('cleanup also runs when required map check fails after native construction')", """passed('cleanup also runs when required map check fails after native construction')
 M._loaded_vendor_pins=lambda require=False:None
 def broken_adapter(c,*,native_factory):
  native_factory(c);raise RuntimeError('adapter construction failed')
 good_adapt=adapt;adapt=broken_adapter
 cand=M.make(c);events.clear();reject(cand.implementation.run);assert events[-1]=='cleanup'
 adapt=good_adapt
 def throwing_build(c):
  child=build(c)
  child.implementation.run=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('native run failed'))
  return child
 normal_build=build
 # Wrap original without recursive name lookup.
 def failing_build(c):
  child=normal_build(c);child.implementation.run=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('native run failed'));return child
 build=failing_build;cand=M.make(c);events.clear();reject(cand.implementation.run);assert events[-1]=='cleanup';build=normal_build
 passed('cleanup on adapter construction failure and native execution failure')""")
s=s.replace("M._snapshot(runtime=True)","M._snapshot(runtime=True)")
s=s.replace("passed('actual canonical checker constructs Candidate before first Gemm inputs access')","passed('actual canonical checker constructs Candidate before first inputs access')")
s=s.replace("assert not forbidden\n", """# Same params but altered indices cannot enter; snapshot recursively immutable.
for c in cs:
 bad=types.SimpleNamespace(id=c.id,params=c.params,_call=types.SimpleNamespace(signature=c._call.signature,indices=dict(c._call.indices,T='float16'),tensors=c._call.tensors))
 assert not M.matches(bad);reject(lambda:M.make(bad))
 frozen=M._frozen_case(M._key(c))
 assert dict(frozen._call.indices)==dict(c._call.indices)
 try:frozen._call.indices['T']='float16'
 except TypeError:pass
 else:raise AssertionError('mutable snapshot')
passed('all seven resolved indices bind independently of report params; immutable adapter snapshots')
assert not forbidden
""")
s=s.replace('# Full real pin check',(O/'extra_checks.txt').read_text()+'\n# Full real pin check')
s=s.replace('original_import=builtins.__import__; forbidden=[]',"original_import=builtins.__import__; forbidden=[]; native_load_attempts=[]\nimport ctypes\ndef audit(event,args):\n if event == 'ctypes.dlopen':\n  native_load_attempts.append(str(args));raise AssertionError('Native library load forbidden in CPU checks')\nsys.addaudithook(audit)")
s=s.replace("'native_library_loads':0","'native_library_loads':len(native_load_attempts),'native_load_attempts':native_load_attempts")
s=s.replace("result={'passed':True","assert not forbidden and not native_load_attempts\nresult={'passed':True")
(O/'cpu_checks.py').write_text(s)
