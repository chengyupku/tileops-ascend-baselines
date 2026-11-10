import fcntl,json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
W=Path('/opt/tileops-all-pairs/workspace')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((P/'representative.json').read_text());e=json.loads((P/'EXECUTION.json').read_text())
assert r['status']=='verified' and r['verification']['checked_outputs']==r['verification']['total_outputs']==1
assert r['inputs_preserved'] and e['frozen_unchanged'] and (P/'guard.exit').read_text().strip()=='0'
f=json.loads((P/'FROZEN.json').read_text());assert all(sha(Path(k))==v for k,v in f['sha256'].items())
(P/'SOURCE_PIN_AFTER.json').write_text(json.dumps({'all_frozen_unchanged':True,'sha256':{k:sha(Path(k)) for k in f['sha256']}},indent=2))
target=W/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer';init=target/'__init__.py';module=target/'integration700_gmm_wgrad_first1.py'
addition="""
# integration700_gmm_wgrad_first1: canonical full-output verified exact key.
from .integration700_gmm_wgrad_first1 import verified_wgrad as _gmm_wgrad_first1_factory
_gmm_wgrad_first1_previous = PROVIDERS.get('GroupedGemmFwdOp')
def _gmm_wgrad_first1_dispatch(case):
    if case.id == 'moe-wgrad-float16':
        return _gmm_wgrad_first1_factory(case)
    if _gmm_wgrad_first1_previous is not None:
        return _gmm_wgrad_first1_previous(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No verified gmm_wgrad_first1 route for this exact key')
PROVIDERS['GroupedGemmFwdOp'] = _gmm_wgrad_first1_dispatch
"""
with open('/tmp/baseline700-provider-ops_transformer.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 before=init.read_bytes();assert b'_gmm_wgrad_first1_dispatch' not in before and not module.exists()
 (P/'registry_before.py').write_bytes(before)
 module.write_bytes((P/module.name).read_bytes());init.write_bytes(before+addition.encode())
 proof={'key':['ops-transformer','GroupedGemmFwdOp','moe-wgrad-float16'],'factory':'benchmarks.ascend_all_pairs.ops_transformer.integration700_gmm_wgrad_first1.verified_wgrad','tested_binding_sha256':sha(P/'remaining_binding.py'),'module':str(module),'module_sha256':sha(module),'registry_before_sha256':hashlib.sha256(before).hexdigest(),'registry_after_sha256':sha(init),'previous_routes_preserved':init.read_bytes().startswith(before),'lock':'/tmp/baseline700-provider-ops_transformer.lock'}
 (P/'REGISTRATION_PROOF.json').write_text(json.dumps(proof,indent=2))
print('exact key registered; all frozen source pins unchanged')
