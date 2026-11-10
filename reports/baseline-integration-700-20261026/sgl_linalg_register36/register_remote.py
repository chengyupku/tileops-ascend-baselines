import sys,pathlib,json,hashlib,types,fcntl,importlib,subprocess
W=pathlib.Path('/opt/tileops-all-pairs/workspace');P=W/'reports/baseline-integration-700-20261026/sgl_linalg_register36';O=P.with_name('sgl_linalg36');P.mkdir(exist_ok=True)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
name='integration700_sgl_linalg_register36';pkg='benchmarks.ascend_all_pairs.sgl';dest=W/'TileOPs/benchmarks/ascend_all_pairs/sgl';scope=json.loads((O/'SCOPE.json').read_text());original=O/'integration700_sgl_linalg36.py';expected='68327a960505a3eb80f6d51dbb1293a6916ad98ea96ca108e785b59be3316ac0'
assert h(original)==expected
for path,digest in json.loads((O/'SOURCE_HASHES.json').read_text()).items():
 p=pathlib.Path(path);p=p if p.is_absolute() else W/p;assert h(p)==digest
# Pure Python stand-ins: never import torch, torch_npu, or load a native library.
for token in ['torch','torch.nn','torch.nn.functional']:
 m=types.ModuleType(token);m.__path__=[];sys.modules[token]=m
sys.modules['torch'].nn=sys.modules['torch.nn'];sys.modules['torch.nn'].functional=sys.modules['torch.nn.functional']
source=pathlib.Path(json.loads((O/'DELIVERY.json').read_text())['source_directory'])
sgl=types.ModuleType('sgl_kernel_npu');sgl.__file__=str(source/'python/sgl_kernel_npu/sgl_kernel_npu/__init__.py');sys.modules['sgl_kernel_npu']=sgl
api=types.ModuleType('benchmarks.api');api.Implementation=lambda run:types.SimpleNamespace(run=run);sys.modules[api.__name__]=api
base=types.ModuleType('benchmarks.ascend_baselines')
class UnsupportedCase(Exception):pass
base.UnsupportedCase=UnsupportedCase
base.Candidate=lambda implementation,api,adaptation,source_sha256,**kw:types.SimpleNamespace(implementation=implementation,api=api,adaptation=adaptation,source_sha256=source_sha256,**kw)
base.source_hashes=lambda *a:None;native_calls=[]
base._owned_native=lambda binary,symbols:native_calls.append({'binary':str(binary),'sha256':h(binary),'symbols':symbols})
sys.modules[base.__name__]=base
sys.path.insert(0,str(W/'TileOPs'))
reg=importlib.import_module('benchmarks.ascend_all_pairs.registry');provider=importlib.import_module(pkg)
append=(O/'provider_registration.append.py').read_text().replace('integration700_sgl_linalg36',name)
# Prove exact scope in proposed append independently of predecessor metadata.
ns={};import ast
for node in ast.parse(append).body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==f'_{name}_keys' for t in node.targets):keys=ast.literal_eval(node.value)
assert {(op,cid) for op,cs in keys.items() for cid in cs}=={(r['operator'],r['case_id']) for r in scope}
with open('/tmp/baseline700-provider-sgl.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 init=dest/'__init__.py';before=init.read_bytes();assert name not in before.decode(),'Already registered; inspect rather than append twice'
 (P/'provider_init.before.py').write_bytes(before)
 importlib.reload(provider);previous=dict(provider.PROVIDERS)
 target=dest/(name+'.py');assert not target.exists();target.write_bytes(original.read_bytes());assert h(target)==expected
 try:
  init.write_bytes(before+b'\n'+append.encode());importlib.invalidate_caches();importlib.reload(provider)
  rows=[]
  for r in scope:
   case=types.SimpleNamespace(id=r['case_id'],params={},_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=r['operator'])))
   candidate=reg.candidate_factory(case,r['baseline'])()
   assert candidate.implementation.run.__module__==pkg+'.'+name
   assert candidate.source_sha256[str(target)]==expected
   rows.append({'key':[r['baseline'],r['operator'],r['case_id']],'factory':pkg+'.'+name+':build','implementation_module':candidate.implementation.run.__module__,'api':candidate.api,'module_sha256':h(target),'native_ownership_stub':native_calls[-1]})
  fallback=[]
  for op in keys:
   case=types.SimpleNamespace(id='__wrong_key__',_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)))
   route=provider.PROVIDERS[op];cells=[c.cell_contents for c in route.__closure__];assert any(v is previous.get(op) for v in cells)
   if previous.get(op) is None:
    try:reg.candidate_factory(case,'sgl-kernel-npu')()
    except UnsupportedCase:fallback.append({'operator':op,'result':'UnsupportedCase; previous absent preserved'})
    else:raise AssertionError('wrong key accepted')
   else:
    # Sentinels show the wrapper forwards its exact previous function.
    old=previous[op];cell=next(c for c in route.__closure__ if c.cell_contents is old);sentinel=object();cell.cell_contents=lambda c:sentinel
    assert reg.candidate_factory(case,'sgl-kernel-npu')() is sentinel;cell.cell_contents=old
    fallback.append({'operator':op,'result':'exact previous callable preserved and delegation exercised'})
  assert all(provider.PROVIDERS[k] is v for k,v in previous.items() if k not in keys)
  assert 'torch_npu' not in sys.modules
  checks={'cpu_only':True,'real_registry_called':True,'candidate_factories_called':len(rows),'rows':rows,'fallback':fallback,'torch_imported':False,'torch_npu_imported':False,'device_libraries_loaded':False,'stub_boundary':['torch and torch.nn.functional','sgl_kernel_npu package','Candidate and Implementation data carriers','_owned_native (binary hash read; nm separately)'],'numerical_reruns':0}
  registration={'host':'server2','registered_count':len(rows),'module':str(target),'module_sha256':h(target),'provider':str(init),'provider_before_sha256':hashlib.sha256(before).hexdigest(),'provider_after_sha256':h(init),'backup':str(P/'provider_init.before.py'),'lock':'/tmp/baseline700-provider-sgl.lock','preserved_prefix':init.read_bytes().startswith(before),'source_migration':'Byte-identical module. __file__ affects provenance hash key only; SOURCE absolute and closure logic unchanged. No relative imports in adapter.','registry_sha256':h(W/'TileOPs/benchmarks/ascend_all_pairs/registry.py'),'rows':rows}
  (P/'DEVICE_FREE_CHECKS.json').write_text(json.dumps(checks,indent=2));(P/'REGISTRATION.json').write_text(json.dumps(registration,indent=2));(P/'registration.append.py').write_text(append)
 except BaseException:
  init.write_bytes(before)
  raise
print(json.dumps({'registered_count':len(rows),'module_sha256':h(target),'fallback_checks':len(fallback)}))
