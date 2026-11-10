import ast,pathlib,json,hashlib,pprint,sys,datetime
T=pathlib.Path(__file__).resolve().parent;W=pathlib.Path('/root/dyq/workspace');R=T.parents[1]
sys.dont_write_bytecode=True;sys.path[:0]=[str(W/'TileOPs/src'),str(W/'.venv/lib/python3.12/site-packages')]
from tileops.manifest import load_manifest,load_adts,load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate
src=(R/'bf16_moe_down_registration3/integration700_bf16_moe_down_registration3.py').read_text();tree=ast.parse(src)
names={'_key','_plain','_canonical','matches','_deepfreeze','_freeze_case','_proc','_guard_identity','_require_guard'}
funcs='\n\n'.join(ast.get_source_segment(src,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names)
rows=[r for r in json.loads((T/'CASES.json').read_text()) if (p:=T/r['key'][2]/'RESULT.json').exists() and json.loads(p.read_text())['status'] in ('verified','verified_pending_parent_registration') and json.loads(p.read_text())['guard_exit']==0]
assert rows
params={tuple(r['key']):json.dumps(r['manifest_params'],sort_keys=True,separators=(',',':')) for r in rows}
plan=entry_plan('MoEGroupedGemmFwdOp',load_manifest()['MoEGroupedGemmFwdOp'],load_adts(),resolve=False)
def plain(v):
 if getattr(v,'adt',None):return {v.kind:plain(v.fields)}
 if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [plain(x) for x in v]
 return v
calls={c.case_id:c for r in load_workloads('MoEGroupedGemmFwdOp') for d in r.get('dtype_cases') or [{}] if (c:=instantiate(plan,r,d)).case_id in {x['key'][2] for x in rows}}
indices={k:plain(calls[k[2]].indices) for k in params}
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
pins=json.loads((T/'FROZEN.json').read_text())['sha256'];pins.update({p:v['current'] for p,v in json.loads((T/'RUNTIME_IDENTITIES.json').read_text())['files'].items()})
for p in [W/'TileOPs/benchmarks/api.py',W/'TileOPs/benchmarks/ascend_baselines.py']:pins[str(p)]=sha(p)
# Exclude orchestration files; runtime dependencies and original source remain frozen.
pins={p:h for p,h in pins.items() if not p.startswith(str(T)) or pathlib.Path(p).name=='adapter.py'}
(T/'REGISTRATION_PINS.json').write_text(json.dumps(pins,indent=2))
head='''"""Exact verified MoE up routes; CPU-lazy, unchanged native cleanup lifetime."""
import ast,dataclasses,hashlib,json,os,types,typing
from pathlib import Path
from types import MappingProxyType
'''
head+='ROOT = Path('+repr(str(T))+')\n'
head+='PIN_HASH = '+repr(sha(T/'REGISTRATION_PINS.json'))+'\n'
head+='CASE_PARAMS = '+pprint.pformat(params)+'\nCASE_INDICES = '+pprint.pformat(indices)+'\n'
body='''
def _pins():
 raw=(ROOT/'REGISTRATION_PINS.json').read_bytes()
 if hashlib.sha256(raw).hexdigest()!=PIN_HASH:raise RuntimeError('Registration pins changed')
 return json.loads(raw)

def _native(case):
 _require_guard()
 vendor='/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/install/vendors/bf16gmm_transformer'
 if os.environ.get('ASCEND_CLASSIC_TRANSFORMER_VENDOR')!=vendor or vendor not in os.environ.get('ASCEND_CUSTOM_OPP_PATH','').split(':'):
  raise RuntimeError('Original vendor environment required; use supplied environment.sh')
 if os.environ.get('ASCEND_HOME_PATH')!='/usr/local/Ascend/cann-9.2.0-beta.2':raise RuntimeError('Original SDK required')
 pins=_pins()
 for p,h in pins.items():
  with open(p,'rb') as f: observed=hashlib.file_digest(f,'sha256').hexdigest()
  if observed!=h:raise RuntimeError('Original dependency changed: '+p)
 adapter=ROOT/'adapter.py';raw=adapter.read_bytes()
 module=types.ModuleType('_verified_moe_up_adapter');module.__file__=str(adapter)
 exec(compile(raw,str(adapter),'exec'),module.__dict__)
 return module.make_candidate(case)

def make(case,baseline='ops-transformer'):
 from benchmarks.ascend_baselines import Candidate,UnsupportedCase
 if not matches(case,baseline):raise UnsupportedCase('Exact verified MoE up key and complete manifest contract required')
 frozen=_freeze_case(case)
 # Recover only the original passive dataclass, avoiding device imports in CPU routing.
 path='/root/dyq/workspace/TileOPs/benchmarks/api.py';raw=Path(path).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=_pins()[path]:raise RuntimeError('Original API changed')
 cls=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='Implementation')
 ns=dict(dataclasses=dataclasses,Callable=typing.Callable,Optional=typing.Optional)
 exec(compile(ast.Module(body=[cls],type_ignores=[]),path,'exec'),ns)
 child=[_native(frozen) if _guard_identity() is not None else None]
 def run(*args,**kwargs):
  _require_guard()
  if not matches(case,baseline):raise UnsupportedCase('Manifest mutated after factory construction')
  if child[0] is None:child[0]=_native(frozen)
  return child[0].implementation.run(*args,**kwargs)
 def cleanup():
  if child[0] is not None:
   value,child[0]=child[0],None
   if value.cleanup is not None:value.cleanup()
 return Candidate(ns['Implementation'](run),'ops-transformer::aclnnGroupedMatmulV5',
  'Verified original MoE physical-psum bridge and original FP32 native products; cleanup retained until caller cleanup.',
  dict(_pins()),cleanup=cleanup)

def wrap(previous):
 def dispatch(case):
  if _key(case) in CASE_PARAMS:return make(case)
  return previous(case)
 dispatch.previous=previous
 return dispatch
'''
module=T/'integration700_moe_up.py';module.write_text(head+funcs+'\n'+body);ast.parse(module.read_text())
(T/'REGISTRY.patch').write_text("# Parent-only serial publication: copy integration700_moe_up.py into ops_transformer package, then append:\nfrom .integration700_moe_up import wrap as _moe_up_wrap\nPROVIDERS['MoEGroupedGemmFwdOp'] = _moe_up_wrap(PROVIDERS.get('MoEGroupedGemmFwdOp'))\n")
print('Prepared keys',list(params))
