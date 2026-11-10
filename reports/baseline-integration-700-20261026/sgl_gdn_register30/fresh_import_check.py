import sys, pathlib, json, hashlib, types, importlib, importlib.abc, fcntl, datetime, difflib
sys.dont_write_bytecode=True
W=pathlib.Path('/opt/tileops-all-pairs/workspace');Q=W/'reports/baseline-integration-700-20261026/sgl_gdn_register30';T=W/'TileOPs';D=T/'benchmarks/ascend_all_pairs/sgl';NAME='integration700_sgl_gdn_register30'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=json.loads((Q/'AUDIT.json').read_text())['rows'];scope=json.loads((Q/'SCOPE.json').read_text());assert {tuple(r['key']) for r in rows}=={tuple(r['key']) for r in scope}
source=Q/'frozen_verified/integration700_sgl_gdn_remote38.py';expected='4154c11fd85ae41eca7878bd9e054124c5c85353c8bf86625e8f4a23048aeabf';assert sha(source)==expected
keys={}
for r in rows:keys.setdefault(r['key'][1],[]).append(r['key'][2])
append='''\n# sgl_gdn_register30: exact verified original complete API bindings, preserve prior routes.
from .integration700_sgl_gdn_register30 import build as _integration700_sgl_gdn_register30_build
_integration700_sgl_gdn_register30_keys = KEYS_LITERAL
def _integration700_sgl_gdn_register30_route(previous, keys):
    def route(case):
        if case.id in keys:
            from pathlib import Path
            import hashlib, sys
            package = Path('/opt/tileops-all-pairs/build/sgl-pristine-20261025T2310/source/python/sgl_kernel_npu/sgl_kernel_npu')
            binary = package / 'lib/libsgl_kernel_npu.so'
            if hashlib.sha256(binary.read_bytes()).hexdigest() != 'c8026d311e1bf56f8bff8a03a1908b420efc421c45980885ded6d69b51c24a6d':
                raise RuntimeError('SGL GDN verified library hash mismatch')
            candidate = _integration700_sgl_gdn_register30_build(case)
            if Path(sys.modules['sgl_kernel_npu'].__file__).resolve() != (package / '__init__.py').resolve():
                raise RuntimeError('SGL GDN resolved outside verified original package')
            return candidate
        if previous is not None:
            return previous(case)
        from benchmarks.ascend_baselines import UnsupportedCase
        raise UnsupportedCase('No SGL provider for this unregistered exact key')
    return route
for _integration700_sgl_gdn_register30_op, _integration700_sgl_gdn_register30_cases in _integration700_sgl_gdn_register30_keys.items():
    PROVIDERS[_integration700_sgl_gdn_register30_op] = _integration700_sgl_gdn_register30_route(PROVIDERS.get(_integration700_sgl_gdn_register30_op), frozenset(_integration700_sgl_gdn_register30_cases))
'''.replace('KEYS_LITERAL',repr(keys))
# Device imports are denied. Only two explicit runtime stubs and the native API stub exist.
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','triton','sgl_kernel_npu','tilelang'):
   raise RuntimeError('Forbidden device import in CPU validation: '+fullname)
sys.meta_path.insert(0,Block());sys.path[:0]=[str(T),str(T/'src'),str(W)]
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_all_pairs import sgl
from benchmarks.ascend_baselines import UnsupportedCase
original_providers=dict(sgl.PROVIDERS)
torch=types.ModuleType('torch');sys.modules['torch']=torch
api=types.ModuleType('benchmarks.api');api.Implementation=lambda fn:types.SimpleNamespace(run=fn);sys.modules['benchmarks.api']=api
package='/opt/tileops-all-pairs/build/sgl-pristine-20261025T2310/source/python/sgl_kernel_npu/sgl_kernel_npu'
native=types.ModuleType('sgl_kernel_npu.fla.chunk');native.__file__=package+'/fla/chunk.py'
exec(compile('def chunk_gated_delta_rule_native(*a, **kw): raise RuntimeError("Numerical execution forbidden in CPU registry test")\ndef chunk_gated_delta_rule_npu(*a, **kw): raise RuntimeError("Out-of-scope l2 API")',native.__file__,'exec'),native.__dict__)
sys.modules[native.__name__]=native
parent=types.ModuleType('sgl_kernel_npu');parent.__file__=package+'/__init__.py';sys.modules[parent.__name__]=parent

def check():
 records=[]
 for r in rows:
  op,cid=r['key'][1:];snap=r['snapshot'];params=dict(snap['parameters']);n=10 if op=='GDNFwdOp' else 7
  case=types.SimpleNamespace(id=cid,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)),params=params,inputs=[types.SimpleNamespace(shape=tuple(params['q'][0]))]*n)
  factory=registry.candidate_factory(case,'sgl-kernel-npu');candidate=factory()
  assert candidate.implementation.run.__module__=='benchmarks.ascend_all_pairs.sgl.'+NAME
  assert candidate.api=='sgl_kernel_npu.fla.chunk.chunk_gated_delta_rule_native'
  assert candidate.source_sha256[str(D/(NAME+'.py'))]==expected
  records.append({'key':r['key'],'registry':str(pathlib.Path(registry.__file__).resolve()),'factory_called':True,'run_module':candidate.implementation.run.__module__,'api':candidate.api,'binding_sha256':expected,'source_hash_count':len(candidate.source_sha256)})
 for op in keys:
  wrong=types.SimpleNamespace(id='__unregistered_scope_probe__',_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)))
  if op not in original_providers:
   try:registry.candidate_factory(wrong,'sgl-kernel-npu')()
   except UnsupportedCase:pass
   else:raise AssertionError('Wrong exact key accepted')
 # Existing linalg and elementwise routes retain object identity.
 for op,factory in original_providers.items():
  if op not in keys:assert sgl.PROVIDERS[op] is factory
 sentinel=object();route=sgl._integration700_sgl_gdn_register30_route(lambda case:sentinel,frozenset())
 assert route(types.SimpleNamespace(id='wrong')) is sentinel
 for spec in ['benchmarks.ascend_all_pairs.sgl.'+NAME+':build','benchmarks.ascend_all_pairs.sgl.'+NAME+'.build']:
  module,fn=spec.split(':',1) if ':' in spec else spec.rsplit('.',1);assert getattr(importlib.import_module(module),fn) is sgl._integration700_sgl_gdn_register30_build
 maps=pathlib.Path('/proc/self/maps').read_text();assert not any(s in maps for s in ['libascendcl.so','libtorch_npu.so','libsgl_kernel_npu.so'])
 return {'passed':True,'scope_count':len(records),'dispatch':records,'wrong_exact_keys_rejected':True,'previous_provider_keys_preserved':list(original_providers),'fallback_identity_preserved':True,'factory_syntaxes':['module:function','module.function'],'device_dynamic_libraries_loaded':False,'numerical_execution':False,'torch_is_stub':True,'native_is_stub':True,'interpreter':sys.executable,'registry_sha256':sha(pathlib.Path(registry.__file__))}


checks=check()
for op in keys:
 wrong=types.SimpleNamespace(id='__unregistered_scope_probe__',_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)))
 try:registry.candidate_factory(wrong,'sgl-kernel-npu')()
 except UnsupportedCase:pass
 else:raise AssertionError('wrong key accepted on fresh import')
checks['fresh_process_disk_import']=True
checks['provider_sha256']=sha(D/'__init__.py')
(Q/'FRESH_IMPORT_CHECKS.json').write_text(json.dumps(checks,indent=2))
print(json.dumps({'fresh_import':True,'scope_count':len(checks['dispatch'])}))
