from pathlib import Path
import fcntl,shutil,hashlib,json,ast,datetime
p=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_dense');d=Path('/opt/tileops-all-pairs/workspace/TileOPs/benchmarks/ascend_all_pairs/ops_transformer');name='integration700_transformer_dense_rope.py';target=d/name
block='''
# integration700_transformer_dense: numerically verified exact YaRN key only.
from .integration700_transformer_dense_rope import rope as _integration700_transformer_dense_rope
_integration700_transformer_dense_old_yarn = PROVIDERS.get('YaRNFwdOp')
def _integration700_transformer_dense_rope_dispatch(case):
    if case.id == '1d-5k-float32':
        return _integration700_transformer_dense_rope(case)
    if _integration700_transformer_dense_old_yarn is not None:
        return _integration700_transformer_dense_old_yarn(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No integration700 transformer_dense registration for this exact key')
PROVIDERS['YaRNFwdOp'] = _integration700_transformer_dense_rope_dispatch
'''
assert json.loads((p/'rope_check_graph_cpu_state/YaRNFwdOp.json').read_text())['status']=='verified'
with open('/tmp/baseline700-provider-ops_transformer.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 src=(d/'__init__.py').read_text();assert '_integration700_transformer_dense_yarn_dispatch' not in src
 shutil.copy2(d/'__init__.py',p/'provider_init.yarn.before.py');assert not target.exists()
 shutil.copy2(p/'integration700_transformer_dense_graph.py',d/'integration700_transformer_dense_graph.py');shutil.copy2(p/name,target);new=src+block;ast.parse(new);tmp=d/'integration700_transformer_dense_init.tmp';tmp.write_text(new);tmp.replace(d/'__init__.py')
 (p/'provider_init.yarn.patch').write_text(block)
 (p/'REGISTRATION_YARN.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'key':['ops-transformer','YaRNFwdOp','1d-5k-float32'],'factory':'benchmarks.ascend_all_pairs.ops_transformer.integration700_transformer_dense_rope.rope','module_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'registered':True,'init_before_sha256':hashlib.sha256(src.encode()).hexdigest(),'init_after_sha256':hashlib.sha256(new.encode()).hexdigest(),'lock':'/tmp/baseline700-provider-ops_transformer.lock'},indent=2))
print('registered exact key; syntax valid; existing provider source preserved as prefix')
