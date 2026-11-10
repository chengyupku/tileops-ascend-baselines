"""Exact ten verified DSA keys, isolated frozen binding loader."""
from pathlib import Path
import hashlib, importlib.util, sys
ROOT = Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/sgl_attention_register10')
KEYS = frozenset(['mla-d512-h64-kv8k-bfloat16', 'mla-d512-h128-kv32k-bfloat16', 'mla-d512-h128-kv48k-bfloat16', 'mla-d512-h64-kv32k-bfloat16', 'mla-d512-h64-kv48k-bfloat16', 'mla-d512-h64-kv64k-bfloat16', 'mla-d576-h128-kv128k-bfloat16', 'mla-d576-h128-kv32k-bfloat16', 'mla-d576-h128-kv64k-bfloat16', 'mla-d576-h128-kv96k-bfloat16'])
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
