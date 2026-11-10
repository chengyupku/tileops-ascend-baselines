import pathlib,json,shutil
P=pathlib.Path(__file__).resolve().parent;R=P.with_name('sgl_attention_recovery32');stage=P/'stage';stage.mkdir(exist_ok=True)
shutil.copyfile(R/'delta03/dsa_adapter.py',stage/'dsa_adapter.py')
keys=[r['case_id'] for r in json.loads((P/'SCOPE.json').read_text())]
root='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/sgl_attention_register10'
module='''"""Exact ten verified DSA keys, isolated frozen binding loader."""
from pathlib import Path
import hashlib, importlib.util, sys
ROOT = Path(ROOT_VALUE)
KEYS = frozenset(KEYS_VALUE)
_MODULE = '_baseline700_sgl_attention_register10_frozen_dsa'
def build(case):
    import benchmarks.ascend_baselines as base
    if case.id not in KEYS:
        raise base.UnsupportedCase('Outside SGL attention register10 exact scope')
    import json
    for item in json.loads((ROOT/'PINNED_LIBRARY.json').read_text()):
        if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError('Pinned original SGL dependency changed: '+item['path'])
    path = ROOT/'dsa_adapter.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != 'bb4b09f8f840241f808c6a5df82bd3122038ec0e9d8e1e70a3247f870c152e59':
        raise RuntimeError('Frozen DSA binding changed')
    saved = base.LIBRARIES
    base.LIBRARIES = ROOT/'libraries'
    try:
        if _MODULE not in sys.modules:
            spec=importlib.util.spec_from_file_location(_MODULE,path)
            module=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[_MODULE]=module
        module=sys.modules[_MODULE]
        if Path(module.__file__).resolve()!=path.resolve():
            raise RuntimeError('Unique frozen binding name collision')
        return module.build(case)
    finally:
        base.LIBRARIES=saved
'''.replace('ROOT_VALUE',repr(root)).replace('KEYS_VALUE',repr(keys))
(stage/'integration700_sgl_attention_register10.py').write_text(module)
append='''
# sgl_attention_register10: exact ten audited full canonical successes.
from .integration700_sgl_attention_register10 import build as _ar10_build, KEYS as _ar10_keys
def _ar10_wrap(previous):
    def route(case):
        if case.id in _ar10_keys:
            return _ar10_build(case)
        if previous is not None:
            return previous(case)
        from benchmarks.ascend_baselines import UnsupportedCase
        raise UnsupportedCase('No SGL provider for this unregistered exact key')
    return route
PROVIDERS['DSADecodeWithKVCacheFwdOp'] = _ar10_wrap(PROVIDERS.get('DSADecodeWithKVCacheFwdOp'))
'''
(stage/'provider_registration.append.py').write_text(append)
