"""Five checked FP8 cases; byte-identical frozen adapter and original TileLang API."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/tilelang_register5')
FROZEN = ROOT / 'attempt02/frozen'
RUNTIME = Path('/opt/tileops-ci/runtime/lib/python3.12/site-packages/tilelang')
SDK = Path('/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_native_deepgemm_deepep/sdk-920b2')
KEYS = frozenset(('FP8QuantFwdOp', cid) for cid in (
    'kv-index-4k-float32', 'kv-index-8k-bfloat16', 'kv-index-8k-float16',
    'kv-index-context-chunk-bfloat16', 'kv-index-context-chunk-float16'))


def inspect_pins():
    pins = json.loads((ROOT / 'attempt02/pins.json').read_text())
    bad = [name for name, digest in pins.items()
           if not Path(name).is_file() or hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest]
    if bad:
        raise RuntimeError('Tested TileLang source/environment changed: ' + repr(bad))
    return {'checked': len(pins), 'mismatches': bad}


def factory(case):
    key = (case._call.signature.name, case.id)
    if key not in KEYS:
        raise NotImplementedError(key)
    inspect_pins()
    # This provider requires the original 0.1.15 runner environment. Never swap a loaded runtime.
    for name in ('tilelang', 'tvm'):
        loaded = sys.modules.get(name)
        if loaded is not None:
            expected = RUNTIME if name == 'tilelang' else RUNTIME / '3rdparty/tvm/python/tvm'
            if Path(loaded.__file__).resolve().parent != expected:
                raise RuntimeError('Use the original TileLang 0.1.15 runner; incompatible loaded ' + name)
    if str(RUNTIME.parent) not in sys.path:
        sys.path.append(str(RUNTIME.parent))
    spec = importlib.util.find_spec('tilelang')
    if spec is None or Path(spec.origin).resolve() != RUNTIME / '__init__.py':
        raise RuntimeError('Original TileLang 0.1.15 must be selected before factory construction')
    os.environ.update(BISHENG_HOME=str(SDK/'tools/bisheng_compiler'),
                      ASCEND_HOME_PATH=str(SDK), ASCEND_TOOLKIT_HOME=str(SDK),
                      CPATH=':'.join(str(SDK/x) for x in ['asc/include','asc','include']))
    spec = importlib.util.spec_from_file_location('_integration700_tilelang_register5_frozen', FROZEN/'impl.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Only the vendor source location changes; its bytes and adapter bytes match the checker.
    module.V = FROZEN/'vendor'
    out = ROOT/'provider_receipts'/key[0]/key[1]
    out.mkdir(parents=True, exist_ok=True)
    return module.factory(case, key[0], out)
