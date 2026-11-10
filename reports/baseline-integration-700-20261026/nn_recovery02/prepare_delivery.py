from pathlib import Path
import json,hashlib,datetime,difflib
P=Path(__file__).resolve().parent;W=Path('/root/dyq/workspace');shared=W/'TileOPs/benchmarks/ascend_all_pairs/ops_nn';packet=P/'registration_packet';packet.mkdir(exist_ok=True)
result=json.loads((P/'RESULT.json').read_text());accepted=[r for r in result['rows'] if r['status']=='verified_unregistered'];keys=sorted((r['operator'],r['case_id']) for r in accepted);name='integration700_nn_recovery02_verified'
source=(P/'adapter.py').read_text().replace('def make(case):','def _make_original(case):',1)
source+='\n\nACCEPTED_KEYS = frozenset('+repr(keys)+')\n'
source+='''\ndef make(case):
 import os
 from benchmarks.ascend_baselines import UnsupportedCase
 key=(case._call.signature.name,case.id)
 if key not in ACCEPTED_KEYS:
  raise UnsupportedCase('No accepted nn_recovery02 exact-key receipt')
 expected='/root/dyq/workspace/' + ('.cache/ascend-classic/native-pilot-20261020/ops-nn/install/vendors/classicpilot_nn' if key[0] in ('GemmFwdOp','GroupedGemmFwdOp') else '.cache/all-pairs/ops_nn/install3/vendors/allpairs_nn')
 if Path(os.environ.get('OPS_NN_VENDOR','')).resolve()!=Path(expected).resolve():
  raise UnsupportedCase('This exact receipt requires its original vendor before NPU initialization: '+expected)
 return _make_original(case)
'''
module=packet/(name+'.py');module.write_text(source)
append='\n# Exact keys from nn_recovery02; pending parent review and registration.\nfrom .'+name+' import make as _nn_recovery02_make, ACCEPTED_KEYS as _nn_recovery02_keys\n'
append+='''_nn_recovery02_prior = {}
def _nn_recovery02_route(op, prior):
    def factory(case):
        if (case._call.signature.name, case.id) in _nn_recovery02_keys:
            return _nn_recovery02_make(case)
        if prior is not None:
            return prior(case)
        from benchmarks.ascend_baselines import UnsupportedCase
        raise UnsupportedCase('No accepted exact key')
    return factory
for _nn_recovery02_op in sorted({key[0] for key in _nn_recovery02_keys}):
    _nn_recovery02_prior[_nn_recovery02_op] = PROVIDERS.get(_nn_recovery02_op)
    PROVIDERS[_nn_recovery02_op] = _nn_recovery02_route(_nn_recovery02_op, PROVIDERS.get(_nn_recovery02_op))
'''
(packet/'registration.append.py').write_text(append)
original=(shared/'__init__.py').read_text();(packet/'provider_init.reference.py').write_text(original)
patch=''.join(difflib.unified_diff([],source.splitlines(True),fromfile='/dev/null',tofile='b/TileOPs/benchmarks/ascend_all_pairs/ops_nn/'+name+'.py'))
patch+=''.join(difflib.unified_diff(original.splitlines(True),(original+append).splitlines(True),fromfile='a/TileOPs/benchmarks/ascend_all_pairs/ops_nn/__init__.py',tofile='b/TileOPs/benchmarks/ascend_all_pairs/ops_nn/__init__.py'))
(packet/'registration.patch').write_text(patch)
# Exact appended routing test, with stub accepted factory; never import device runtime.
import sys,types
fq='benchmarks.ascend_all_pairs.ops_nn.'+name;stub=types.ModuleType(fq);stub.make=lambda case:'expected-factory';stub.ACCEPTED_KEYS=frozenset(keys);sys.modules[fq]=stub
prior=lambda case:'preserved-prior';providers={op:prior for op,_ in keys};providers['unrelated_sentinel']=prior;ns=dict(PROVIDERS=providers,__package__='benchmarks.ascend_all_pairs.ops_nn');exec(compile(append,'registration.append.py','exec'),ns)
for op,cid in keys:
 c=types.SimpleNamespace(id=cid,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)));assert providers[op](c)=='expected-factory';c.id='outside-scope';assert providers[op](c)=='preserved-prior'
assert providers['unrelated_sentinel'] is prior
(packet/'DEVICE_FREE_CHECKS.json').write_text(json.dumps(dict(exact_keys=len(keys),prior_routes_preserved=True,device_imported=False,registered=False),indent=2))
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
maps=[]
for r in accepted:
 cp=json.loads(Path(r['checker_receipt']).read_text());maps.append(dict(key=[r['baseline'],r['operator'],r['case_id']],formal_entry='benchmarks.ascend_all_pairs.ops_nn.'+name+'.make',registered=False,private_factory=r['factory'],checker_receipt=r['checker_receipt'],guard_receipt=r['guard_receipt'],source_sha256=cp.get('source_sha256'),frozen_module_sha256=sha(module)))
started=json.loads((P/'STARTED.json').read_text())['utc'];now=datetime.datetime.now(datetime.timezone.utc);seconds=(now-datetime.datetime.fromisoformat(started)).total_seconds()
guards=[]
for f in P.glob('*/GUARD_RESULT.json'):
 if not (f.parent/'COMMAND.json').exists():continue
 g=json.loads(f.read_text());guards.append(dict(path=str(f),**g))
delivery=dict(updated_at=now.isoformat(),host='local',owner='baseline700-20261026/nn_recovery02',assigned=23,completed=0,verified_unregistered=len(accepted),actual_wall_s=seconds,registration_allowed=False,entry_mappings=maps,unresolved=[r for r in result['rows'] if r['status']!='verified_unregistered'],guards=guards,changed_files=[str(x) for x in packet.iterdir()],source_compare=[str(P/'SOURCE_COMPARE.json'),str(P/'OTHER_SOURCE_COMPARE.json')],recipe='Existing batch COMMAND.json records exact guarded commands. DO NOT rerun passed numerical checks for registration. Parent must review frozen module, acquire /tmp/baseline700-provider-ops_nn.lock, reread live __init__.py and back it up, add module and append exact-key routing without restoring reference snapshot; release lock before any NPU work. Device-free routing verification is supplied.',patch=str(packet/'registration.patch'),provider_snapshot_is_reference_only=True,repeated_success_numerical_runs=0)
(P/'DELIVERY.json').write_text(json.dumps(delivery,indent=2));print('packet',len(keys),'keys')
