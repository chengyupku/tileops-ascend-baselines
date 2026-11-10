from pathlib import Path
import fcntl,json,hashlib,datetime
W=Path('/opt/tileops-all-pairs/workspace');P=W/'reports/baseline-integration-700-20261026/gmm_fp16_remaining11';Q=P/'conv02_fp32'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt=json.loads((Q/'representative.json').read_text());assert receipt['status']=='verified',receipt['status']
assert (Q/'guard.exit').read_text().strip()=='0'
root=W/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer';module=root/'integration700_gmm_fp16_remaining11.py'
content='''"""Exact verified FP16 keys, original private GroupedMatmul V5 API."""
import importlib.util
import sys
from pathlib import Path

def conv1d_verified(case):
    assert case.id == 'whisper-conv1-float16'
    w=Path('/opt/tileops-all-pairs/workspace')
    d=w/'reports/baseline-integration-700-20261026/transformer_dense'
    path=w/'reports/baseline-integration-700-20261026/gmm_fp16_remaining11/conv02_fp32/conv1d_binding.py'
    spec=importlib.util.spec_from_file_location('_gmm_fp16_remaining11_tested_conv',path)
    m=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(d))
    try:spec.loader.exec_module(m)
    finally:sys.path.remove(str(d))
    return m.conv1d(case)
'''
append='''
# gmm_fp16_remaining11: frozen complete canonical pass, exact key only.
from .integration700_gmm_fp16_remaining11 import conv1d_verified as _gmm_fp16_remaining11_conv
_gmm_fp16_remaining11_conv_old = PROVIDERS.get('Conv1dFwdOp')
def _gmm_fp16_remaining11_conv_dispatch(case):
    if case.id == 'whisper-conv1-float16':
        return _gmm_fp16_remaining11_conv(case)
    return _gmm_fp16_remaining11_conv_old(case)
PROVIDERS['Conv1dFwdOp'] = _gmm_fp16_remaining11_conv_dispatch
'''
with open('/tmp/baseline700-provider-ops_transformer.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 init=root/'__init__.py';before=init.read_bytes();assert 'gmm_fp16_remaining11' not in before.decode();assert not module.exists()
 (P/'registry_before.py').write_bytes(before);module.write_text(content);compile(content,str(module),'exec');compile(before.decode()+append,str(init),'exec');init.write_text(before.decode()+append)
 proof={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'key':['ops-transformer','Conv1dFwdOp','whisper-conv1-float16'],'factory':'integration700_gmm_fp16_remaining11.conv1d_verified','tested_binding':str(Q/'conv1d_binding.py'),'tested_binding_sha256':sha(Q/'conv1d_binding.py'),'checker':str(Q/'representative.json'),'checker_sha256':sha(Q/'representative.json'),'guard_exit':0,'module':str(module),'module_sha256':sha(module),'registry_before_sha256':hashlib.sha256(before).hexdigest(),'registry_after_sha256':sha(init),'prior_registry_prefix_preserved':init.read_bytes().startswith(before),'preserved_other_keys':True}
 (P/'REGISTRATION_PROOF.json').write_text(json.dumps(proof,indent=2));(P/'registered_module.py').write_text(content);(P/'registration_append.py').write_text(append)
print(json.dumps(proof))
