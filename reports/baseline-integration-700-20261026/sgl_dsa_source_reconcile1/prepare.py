from pathlib import Path
import json
P=Path(__file__).resolve().parent
C=json.loads((P/'checker.original.json').read_text())
module='''"""Exact DSA source-race reconciliation; unchanged frozen whole-API binding."""
from pathlib import Path
import hashlib, types, json
ROOT = Path(__file__).resolve().parents[4] / 'reports/baseline-integration-700-20261026/sgl_dsa_source_reconcile1'
KEYS = frozenset(['mla-d512-h128-kv64k-bfloat16'])
PARAMETERS = PARAMS
PINS = PINS_VALUE
ADAPTER_SHA256 = 'bb4b09f8f840241f808c6a5df82bd3122038ec0e9d8e1e70a3247f870c152e59'
def build(case):
    import benchmarks.ascend_baselines as base
    if case.id not in KEYS or json.loads(json.dumps(case.params)) != PARAMETERS:
        raise base.UnsupportedCase('Outside canonical DSA source reconciliation exact key/parameters')
    for relative, expected in PINS.items():
        path = ROOT/'libraries/sgl-kernel-npu'/relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Pinned original SGL dependency changed: '+str(path))
    path = ROOT/'dsa_adapter.py'
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != ADAPTER_SHA256:
        raise RuntimeError('Frozen DSA binding changed')
    saved = base.LIBRARIES
    base.LIBRARIES = ROOT/'libraries'
    try:
        # Execute only the just-hashed bytes, independent of generic module/bytecode caches.
        module = types.ModuleType('_baseline700_sgl_dsa_source_reconcile1_frozen')
        module.__file__ = str(path)
        exec(compile(source, str(path), 'exec'), module.__dict__)
        return module.build(case)
    finally:
        base.LIBRARIES = saved
'''
r=json.loads((P/'REMOTE_READ.json').read_text());pins={}
for x in r:
 if '/libraries/sgl-kernel-npu/' in x['path'] and not x.get('missing'):
  pins[x['path'].split('/libraries/sgl-kernel-npu/')[1]]=x['sha256']
assert len(pins)==3
module=module.replace('PARAMS',repr(C['case_snapshot']['parameters'])).replace('PINS_VALUE',repr(pins))
(P/'integration700_sgl_dsa_source_reconcile1.py').write_text(module)
append='''
# sgl_dsa_source_reconcile1: canonical full-output pass, historical source race reconciled.
from .integration700_sgl_dsa_source_reconcile1 import build as _dsa_sr1_build, KEYS as _dsa_sr1_keys
def _dsa_sr1_wrap(previous):
    def route(case):
        if case.id in _dsa_sr1_keys:
            return _dsa_sr1_build(case)
        if previous is not None:
            return previous(case)
        from benchmarks.ascend_baselines import UnsupportedCase
        raise UnsupportedCase('No SGL provider for this unregistered exact key')
    return route
PROVIDERS['DSADecodeWithKVCacheFwdOp'] = _dsa_sr1_wrap(PROVIDERS.get('DSADecodeWithKVCacheFwdOp'))
'''
(P/'provider_registration.append.py').write_text(append)
print('prepared',list(pins))
