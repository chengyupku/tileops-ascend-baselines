from pathlib import Path
import json,hashlib,fcntl,ast,sys,types,os,importlib
sys.dont_write_bytecode=True
P=Path(__file__).resolve().parent;W=Path('/root/dyq/workspace');N=W/'TileOPs/benchmarks/ascend_all_pairs/ops_nn'
rows=[json.loads(p.read_text()) for p in sorted(P.glob('case*/FROZEN.json'))];assert rows
keys=sorted(('GemmFwdOp',r['case_id']) for r in rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
h=sha(P/'adapter.py');assert h=='037f902f6779aa1d84dad1d5d04777e6754cec5e9b3d653b7339ac5f0dd200d7'
wrapper=(W/'reports/baseline-integration-700-20261026/nn_registration02/integration700_nn_registration02.py').read_text()
wrapper=wrapper.replace('nn_registration02','nn_dense_resume8')
start=wrapper.index('_KEYS = ');end=wrapper.index('\n_FROZEN',start);wrapper=wrapper[:start]+'_KEYS = frozenset('+repr(keys)+')'+wrapper[end:]
start=wrapper.index('_FROZEN = ');end=wrapper.index('\n_SHA256',start);wrapper=wrapper[:start]+'_FROZEN = Path('+repr(str(P/'adapter.py'))+')'+wrapper[end:]
append='''
# Exact numerical receipts: baseline-integration-700-20261026/nn_dense_resume8
from . import integration700_nn_dense_resume8 as _nn_dense_resume8
_nn_dense_resume8_prior = PROVIDERS.get('GemmFwdOp')
def _nn_dense_resume8_route(case, _prior=_nn_dense_resume8_prior):
    if (case._call.signature.name, case.id) in _nn_dense_resume8._KEYS:
        return _nn_dense_resume8.make(case)
    if _prior is not None:
        return _prior(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No exact key')
PROVIDERS['GemmFwdOp'] = _nn_dense_resume8_route
'''
with open('/tmp/baseline700-provider-ops_nn.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 old=(N/'__init__.py').read_text();assert '_nn_dense_resume8_route' not in old
 (P/'provider_before.py').write_text(old);ast.parse(old+append);ast.parse(wrapper)
 (P/'integration700_nn_dense_resume8.py').write_text(wrapper)
 (N/'integration700_nn_dense_resume8.py').write_text(wrapper)
 # Import existing CPU-safe disk registry first, then reload the actually written registry.
 sys.path.insert(0,str(W/'TileOPs'))
 m=importlib.import_module('benchmarks.ascend_all_pairs.ops_nn');prior=dict(m.PROVIDERS)
 (N/'__init__.py').write_text(old+append)
 try:
  m=importlib.reload(m)
  registry=importlib.import_module('benchmarks.ascend_all_pairs.registry')
  # Device constructors explicitly stubbed; execute actual frozen adapter factory only.
  assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
  sys.modules['torch']=types.ModuleType('torch')
  graph=importlib.import_module('benchmarks.ascend_all_pairs.ops_nn.graph');common=importlib.import_module('benchmarks.ascend_all_pairs.ops_nn.common')
  calls=[]
  graph.GraphCall=lambda library: calls.append(library) or types.SimpleNamespace(stub=True)
  common.candidate=lambda binding,run,api,*args: ('stub-constructor',api,binding.stub)
  os.environ['OPS_NN_VENDOR']=str(m._nn_dense_resume8._VENDOR);os.environ['ASCEND_CUSTOM_OPP_PATH']=str(m._nn_dense_resume8._VENDOR)
  for op,cid in keys:
   case=types.SimpleNamespace(id=cid,params={},_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)))
   assert registry.candidate_factory(case,'ops-nn')()==('stub-constructor','MatMulV3',True)
  unknown=types.SimpleNamespace(id='__unregistered__',_call=types.SimpleNamespace(signature=types.SimpleNamespace(name='GemmFwdOp')))
  assert m._nn_dense_resume8_route(unknown,_prior=lambda c:'original-fallback')=='original-fallback'
  assert m._nn_dense_resume8_route.__defaults__[0] is m._nn_dense_resume8_prior
  def outcome(fn):
   try:return ('return',repr(fn()))
   except Exception as exc:return (type(exc).__name__,str(exc))
  assert outcome(lambda:m._nn_dense_resume8_prior(unknown))==outcome(lambda:registry.candidate_factory(unknown,'ops-nn')())
  assert all(m.PROVIDERS[k].__code__.co_code==v.__code__.co_code for k,v in prior.items() if k!='GemmFwdOp' and hasattr(v,'__code__'))
  os.environ['OPS_NN_VENDOR']='wrong'
  try:m._nn_dense_resume8.make(case)
  except ValueError:pass
  else:raise AssertionError('Wrong vendor accepted')
  assert 'torch_npu' not in sys.modules
 except BaseException:
  (N/'__init__.py').write_text(old);raise
 checks={'actual_disk_registry_import':True,'exact_keys':len(keys),'GraphCall_stub_calls':len(calls),'torch_module_stub':True,'candidate_constructor_stub':True,'torch_npu_imported':False,'unknown_key_original_fallback':True,'wrong_vendor_rejected':True,'numerical_claim':False}
 (P/'DEVICE_FREE_CHECKS.json').write_text(json.dumps(checks,indent=2))
 (P/'REGISTRATION.json').write_text(json.dumps({'keys':keys,'registered':len(keys),'parent_accepted_added':0,'append_only':True,'old_sha256':hashlib.sha256(old.encode()).hexdigest(),'new_sha256':sha(N/'__init__.py'),'module_sha256':sha(N/'integration700_nn_dense_resume8.py'),'lock':'/tmp/baseline700-provider-ops_nn.lock'},indent=2))
print('registered',len(keys),'CPU construction stubs only')
