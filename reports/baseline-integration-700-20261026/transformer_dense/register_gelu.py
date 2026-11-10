from pathlib import Path
import fcntl,shutil,hashlib,json,ast,datetime
p=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_dense');d=Path('/opt/tileops-all-pairs/workspace/TileOPs/benchmarks/ascend_all_pairs/ops_transformer');name='integration700_transformer_dense_gelu.py';target=d/name
block='''
# integration700_transformer_dense: numerically verified exact key only.
from .integration700_transformer_dense_gelu import gelu as _integration700_transformer_dense_gelu
_integration700_transformer_dense_old_gelu = PROVIDERS.get('GeluTanhAndMulFwdOp')
def _integration700_transformer_dense_gelu_dispatch(case):
    if case.id == 'llama-8b-ffn-prefill-float16':
        return _integration700_transformer_dense_gelu(case)
    if _integration700_transformer_dense_old_gelu is not None:
        return _integration700_transformer_dense_old_gelu(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No integration700 transformer_dense registration for this exact key')
PROVIDERS['GeluTanhAndMulFwdOp'] = _integration700_transformer_dense_gelu_dispatch
'''
assert json.loads((p/'checker').read_text())['status']=='verified'
with open('/tmp/baseline700-provider-ops_transformer.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 src=(d/'__init__.py').read_text();assert '# integration700_transformer_dense:' not in src
 shutil.copy2(d/'__init__.py',p/'provider_init.before.py');assert not target.exists()
 shutil.copy2(p/name,target);new=src+block;ast.parse(new);tmp=d/'integration700_transformer_dense_init.tmp';tmp.write_text(new);tmp.replace(d/'__init__.py')
 (p/'provider_init.patch').write_text(block)
 (p/'REGISTRATION.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'key':['ops-transformer','GeluTanhAndMulFwdOp','llama-8b-ffn-prefill-float16'],'factory':'benchmarks.ascend_all_pairs.ops_transformer.integration700_transformer_dense_gelu.gelu','module_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'registered':True,'init_before_sha256':hashlib.sha256(src.encode()).hexdigest(),'init_after_sha256':hashlib.sha256(new.encode()).hexdigest(),'lock':'/tmp/baseline700-provider-ops_transformer.lock'},indent=2))
print('registered exact key; syntax valid; existing provider source preserved as prefix')
