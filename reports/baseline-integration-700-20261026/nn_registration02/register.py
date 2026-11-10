import pathlib,json,hashlib,fcntl,sys,os,types,importlib,ast
sys.dont_write_bytecode=True
W=pathlib.Path('/root/dyq/workspace');D=W/'reports/baseline-integration-700-20261026';P=D/'nn_registration02';N=W/'TileOPs/benchmarks/ascend_all_pairs/ops_nn'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(n,v):(P/n).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
rows=json.loads((P/'AUDIT.json').read_text())['rows'];assert len(rows)==7 and not json.loads((P/'ALERTS.json').read_text())
with open('/tmp/baseline700-provider-ops_nn.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 acc={tuple(r['key']) for r in json.loads((D/'ACCEPTED.json').read_text())['rows']};rows=[r for r in rows if tuple(r['key']) not in acc]
 assert len(rows)==7,'Acceptance changed; review before registration'
 old=(N/'__init__.py').read_text();assert '_nn_registration02_route' not in old
 (P/('provider_before_'+hashlib.sha256(old.encode()).hexdigest()+'.py')).write_text(old)
 h=rows[0]['adapter_sha256'];frozen=P/('adapter_'+h+'.py');data=pathlib.Path(rows[0]['actual_tested_factory'].removesuffix(':make')).read_bytes()
 assert hashlib.sha256(data).hexdigest()==h
 if frozen.exists():assert frozen.read_bytes()==data
 else:frozen.write_bytes(data);frozen.chmod(0o444)
 keys=sorted(tuple(r['key'][1:]) for r in rows)
 # Loader is private; the only shared edit is the provider's exact-key append.
 wrapper=P/'integration700_nn_registration02.py'
 wrapper.write_text('''"""Reviewed exact Gemm keys; delegate unchanged tested adapter bytes."""
import hashlib
import os
from pathlib import Path
from types import ModuleType
_KEYS = frozenset('''+repr(keys)+''')
_FROZEN = Path('''+repr(str(frozen))+''')
_SHA256 = '''+repr(h)+'''
_VENDOR = Path('/root/dyq/workspace/.cache/ascend-classic/native-pilot-20261020/ops-nn/install/vendors/classicpilot_nn')
def _load():
    data = _FROZEN.read_bytes()
    if hashlib.sha256(data).hexdigest() != _SHA256:
        raise RuntimeError('Reviewed frozen adapter changed')
    module = ModuleType('integration700_nn_registration02_frozen')
    module.__file__ = str(_FROZEN)
    exec(compile(data, str(_FROZEN), 'exec'), module.__dict__)
    return module

def make(case):
    from benchmarks.ascend_baselines import UnsupportedCase
    if (case._call.signature.name, case.id) not in _KEYS:
        raise UnsupportedCase('No nn_registration02 exact receipt')
    if Path(os.environ.get('OPS_NN_VENDOR', '')).resolve() != _VENDOR or os.environ.get('ASCEND_CUSTOM_OPP_PATH') != str(_VENDOR):
        raise UnsupportedCase('Select recorded classicpilot_nn vendor before runtime initialization: ' + str(_VENDOR))
    return _load().make(case)
''')
 append='''
# Exact registration: baseline-integration-700-20261026/nn_registration02
from importlib.util import spec_from_file_location as _nn_registration02_spec, module_from_spec as _nn_registration02_module
_nn_registration02_s = _nn_registration02_spec('integration700_nn_registration02', '''+repr(str(wrapper))+''')
_nn_registration02 = _nn_registration02_module(_nn_registration02_s)
_nn_registration02_s.loader.exec_module(_nn_registration02)
_nn_registration02_prior = PROVIDERS.get('GemmFwdOp')
def _nn_registration02_route(case, _prior=_nn_registration02_prior):
    if (case._call.signature.name, case.id) in _nn_registration02._KEYS:
        return _nn_registration02.make(case)
    if _prior is not None:
        return _prior(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No accepted exact key')
PROVIDERS['GemmFwdOp'] = _nn_registration02_route
'''
 ast.parse(old+append);ast.parse(wrapper.read_text());ast.parse(data)
 # Import only CPU-safe provider and use sentinel candidate builders.
 sys.path.insert(0,str(W/'TileOPs'))
 m=importlib.import_module('benchmarks.ascend_all_pairs.ops_nn');prior=dict(m.PROVIDERS)
 (N/'__init__.py').write_text(old+append)
 try:
  exec(compile(append,str(N/'__init__.py'),'exec'),m.__dict__)
  registry=importlib.import_module('benchmarks.ascend_all_pairs.registry')
  assert m._nn_registration02_prior is prior['GemmFwdOp']
  assert all(m.PROVIDERS[k] is v for k,v in prior.items() if k!='GemmFwdOp')
  actual=m._nn_registration02._load();assert actual.__file__==str(frozen) and actual.make.__code__.co_filename==str(frozen)
  m._nn_registration02._load=lambda:types.SimpleNamespace(make=lambda c:('frozen-sentinel',c.id))
  os.environ['OPS_NN_VENDOR']=str(m._nn_registration02._VENDOR);os.environ['ASCEND_CUSTOM_OPP_PATH']=str(m._nn_registration02._VENDOR)
  for r in rows:
   op,cid=r['key'][1:];case=types.SimpleNamespace(id=cid,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)))
   assert registry.candidate_factory(case,'ops-nn')()==('frozen-sentinel',cid)
   p=r['case_snapshot']['parameters'];dtype=p['d'][1];tol=.001 if dtype=='float16' else .016;k=p['a'][0][1] if not p['trans_a'] else p['a'][0][0];gemv=not p['trans_a'] and ((p['d'][0][0]==1 and p['trans_b']) or (p['d'][0][1]==1 and not p['trans_b']))
   assert r['input_seed']==int.from_bytes(hashlib.sha256(f'{op}::{cid}'.encode()).digest()[:4],'big')
   r.update(status='registered_pending_parent_acceptance',registered=True,module=str(frozen),module_sha256=h,actual_factory=str(frozen)+':make',route_module=str(wrapper),route_module_sha256=sha(wrapper),route_validation={'formal_registry_exact_key':True,'frozen_load_hash_verified':True,'stub_candidate_only':True,'old_fallback_identity_preserved':True,'device_imports':False},canonical_tolerance={'rtol':tol,'atol':tol*max(1,k/2048) if gemv else tol,'source':'TileOPs/workloads/gemm.py:GemmWorkload.verification'})
  for op,cid in [('GemmFwdOp','__unknown__'),('OtherOp',keys[0][1])]:
   case=types.SimpleNamespace(id=cid,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)));assert m._nn_registration02_route(case,_prior=lambda c:'prior')=='prior'
  os.environ.pop('OPS_NN_VENDOR')
  try:m._nn_registration02.make(types.SimpleNamespace(id=keys[0][1],_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=keys[0][0]))))
  except ValueError:pass
  else:raise AssertionError('Wrong vendor accepted')
  assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
 except BaseException:
  (N/'__init__.py').write_text(old);raise
 save('REGISTRATION.json',{'rows':rows,'provider':str(N/'__init__.py'),'provider_sha256':sha(N/'__init__.py'),'previous_provider_sha256':hashlib.sha256(old.encode()).hexdigest(),'append_only':True,'lock':'/tmp/baseline700-provider-ops_nn.lock'})
 save('RECOMMENDED_ACCEPT.json',{'rows':rows,'parent_review_required':True,'ledger_modified':False})
 save('DEVICE_FREE_CHECKS.json',{'exact_keys':len(rows),'unknown_key_fallback':True,'wrong_operator_fallback':True,'unrelated_provider_identity_preserved':True,'wrong_vendor_rejected':True,'frozen_bytes_hash_equal':True,'torch_imported':False,'torch_npu_imported':False,'numeric_reruns':0})
print('registered',len(rows))
