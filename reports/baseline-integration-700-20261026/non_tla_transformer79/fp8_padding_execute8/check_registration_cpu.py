import importlib.util,json,sys,types
from pathlib import Path
P=Path(__file__).resolve().parent
# Only interface construction is tested: device imports and input access forbidden.
class Candidate:
 def __init__(self,*args,**kwargs): self.args=args
class UnsupportedCase(Exception): pass
m=types.ModuleType('benchmarks.ascend_baselines');m.Candidate=Candidate;m.UnsupportedCase=UnsupportedCase
sys.modules['benchmarks']=types.ModuleType('benchmarks');sys.modules['benchmarks.ascend_baselines']=m
spec=importlib.util.spec_from_file_location('private_registration',P/'fp8_registration.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class Case:
 def __init__(self,r):
  self.id=r['key'][2];self.params=r['snapshot']['parameters'];self._call=types.SimpleNamespace(signature=types.SimpleNamespace(name='GemmFP8FwdOp'),tensors=r['snapshot']['declared_tensors'])
 @property
 def inputs(self):raise AssertionError('construction accessed inputs')
rows=json.loads((P/'FP8_VERIFIED_CASES.json').read_text());providers={'GemmFP8FwdOp':lambda c:'fallback'};mod.install(providers)
for r in rows:assert isinstance(providers['GemmFP8FwdOp'](Case(r)),Candidate)
assert providers['GemmFP8FwdOp'](types.SimpleNamespace(id='outside-scope'))=='fallback'
c=Case(rows[0]);c.params={'changed':True}
try:providers['GemmFP8FwdOp'](c);raise AssertionError('changed contract accepted')
except UnsupportedCase:pass
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
(P/'REGISTRATION_CPU_CHECK.json').write_text(json.dumps({'contracts':len(rows),'exact_routes':True,'changed_contract_rejected':True,'fallback_preserved':True,'no_inputs_access':True,'no_torch_imports':True},indent=2))
print('CPU registration checks passed',len(rows))
