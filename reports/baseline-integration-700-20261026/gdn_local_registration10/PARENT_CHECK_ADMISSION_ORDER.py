"""CPU-only orchestration check. No numerical/runtime validation is claimed."""
import pathlib,sys,importlib.util,importlib.abc,json,types,tempfile
P=pathlib.Path(__file__).resolve().parent
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','acl'):raise AssertionError('No device import allowed')
sys.meta_path.insert(0,Deny());sys.path.insert(0,'/root/dyq/workspace/TileOPs')
spec=importlib.util.spec_from_file_location('reviewed_gdn10',P/'integration700_gdn_local_registration10.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
key=next(iter(m.CASE_PARAMS));c=types.SimpleNamespace(id=key[2],params=json.loads(m.CASE_PARAMS[key]),_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=key[1])))
identity=('boot',1,2,'request',3,4,'owner');events=[];current=identity
m._guard_identity=lambda:current;m._snapshot=lambda **kwargs: events.append('pins_recheck') or {};m._environment=lambda:events.append('environment');m._loaded_vendor_pins=lambda **kwargs:None
with tempfile.TemporaryDirectory(dir=P) as td:
 def prepare(k):
  events.append('admission_before_inputs')
  def build(frozen):
   events.append('build')
   return types.SimpleNamespace(implementation=types.SimpleNamespace(run=lambda *a,**kw:events.append('numeric_stub') or 'stub'),cleanup=lambda:events.append('cleanup'))
  return {'direct_acl':types.SimpleNamespace(require_guard=lambda:events.append('guard')),'binding':types.SimpleNamespace(build=build)},pathlib.Path(td),identity
 m._prepare_admission=prepare
 x=m.make(c);assert 'admission_before_inputs' in events and 'build' not in events
 events.append('checker_allocates_inputs_and_reference');assert x.implementation.run('fake')=='stub'
 assert events.index('admission_before_inputs')<events.index('checker_allocates_inputs_and_reference')<events.index('build')
 assert events.count('admission_before_inputs')==1 and events.count('cleanup')==1
 events.clear();y=m.make(c);current=('different',)
 try:y.implementation.run('fake')
 except RuntimeError as e:assert 'different guard'in str(e)
 else:raise AssertionError('Guard change accepted')
 assert 'build'not in events
 current=identity;events.clear();z=m.make(c)
 def changed(**kwargs):raise RuntimeError('dependency changed')
 m._snapshot=changed
 try:z.implementation.run('fake')
 except RuntimeError as e:assert 'dependency changed'in str(e)
 else:raise AssertionError('Changed dependencies accepted')
 assert 'build'not in events
assert not any(x in sys.modules for x in ('torch','torch_npu','acl'))
# Verify actual checker orders its factory before first GDN case.inputs use.
source=pathlib.Path('/root/dyq/workspace/TileOPs/scripts/ascend_pair_check.py').read_text()
factory=source.index('candidate = candidate_factory(case, args.library)()')
assert source.index('reference = cpu_reference(torch, reference, case.inputs)')>factory
out=dict(passed=True,checks=['admission before checker input/reference allocation','single admission not double charged','guard identity change rejects before nativebuild','dependency changes reject before nativebuild','cleanup after call','actual checker factory precedes GDN inputs use'],device_imports=0,numerical_runs=0)
(P/'PARENT_ADMISSION_ORDER_CHECK.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
