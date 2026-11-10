"""Exact verified Where binding; imports remain device-free until factory().

The deployment is intentionally server1-specific. The original private adapter,
AST loader and runtime are hash-pinned, reused without editing or monkeypatching
public registry state. Other cases retain the historical provider.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types

HERE = Path(__file__).resolve().parent
_CONFIG = None
_CONTEXT = None


def config():
    return {'workspace': '/opt/tileops-all-pairs/workspace', 'vendor': '/opt/tileops-all-pairs/workspace/ascend_baselines/tilelang-ascend', 'compiler': '/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_tla_compiler', 'runtime': '/opt/tileops-all-pairs/workspace/.cache/all-pairs/full_rollout_20261024/remote_ready/server1/group1/recovery_resume25/tla_supplemental_1858/runtime_env2/runtime/package', 'private': '/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/tla_simple_tilelang/frozen_gemm7', 'output_root': '/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/tla_simple_tilelang/formal_gemm7', 'pins': {'/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/tla_simple_tilelang/frozen_gemm7/adapter.py': 'abfbddca187874043c32c239cb7ec4e169fe40141cbfc5019a551244c2112c71', '/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/tla_simple_tilelang/frozen_gemm7/run_case.py': '173a62e2d92430b4c0f93d54a2f96d59227c60e9ce6b5776b619b9c93eaea693', '/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/tla_simple_tilelang/frozen_gemm7/private_tla/native.py': 'c8d3bf38e23631a4174863b71552c7f7dc0fef0f3d677d7b9e08c953894e7666', '/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/tla_simple_tilelang/frozen_gemm7/private_tla/deployment.py': '3907f1e9c364cad2f0f699d69ce81d410c87afc17a0476dcdbfed13414ff1fb1', '/opt/tileops-all-pairs/workspace/ascend_baselines/tilelang-ascend/examples/gemm/example_gemm.py': '9f7d3855e6ed2ac0a936a5dfb9f4edb50db9a169396f740d68b7ddd52f658f59', '/opt/tileops-all-pairs/workspace/ascend_baselines/tilelang-ascend/examples/batch_gemm/batch_gemm.py': '969609ff44ae23316b6199ee2b7942658543336137d4c66249c897c6686e511a'}, 'keys': [['GemmFwdOp', 'ds-v3-m16-mlp-up-bfloat16'], ['BmmFwdOp', 'square-256-bfloat16']], 'params': {'GemmFwdOp/ds-v3-m16-mlp-up-bfloat16': {'a': [[16, 7168], 'bfloat16'], 'b': [[4096, 7168], 'bfloat16'], 'd': [[16, 4096], 'bfloat16'], 'trans_a': False, 'trans_b': True}, 'BmmFwdOp/square-256-bfloat16': {'a': [[32, 256, 256], 'bfloat16'], 'b': [[32, 256, 256], 'bfloat16'], 'd': [[32, 256, 256], 'bfloat16']}}}

def matches(op, case):
    return [op, case.id] in config()["keys"]


def inspect_pins():
    """CPU-only identity check; no shared library loading."""
    mismatches = []
    for path, expected in config()['pins'].items():
        p = Path(path)
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            mismatches.append(path)
    if mismatches:
        raise RuntimeError('Verified Where deployment changed: ' + ', '.join(mismatches))


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _prepare():
    global _CONTEXT
    if _CONTEXT is not None:
        return _CONTEXT
    inspect_pins()
    c = config()
    W, V, C, R, P = (Path(c[k]) for k in ('workspace', 'vendor', 'compiler', 'runtime', 'private'))
    if any(n.split('.')[0] in ('tilelang', 'tvm', 'cython_wrapper') for n in sys.modules):
        raise RuntimeError('Verified Where requires a fresh process before TileLang/TVM import')
    root = Path(os.environ.get('TLA_VERIFIED_WHERE_OUTPUT', c['output_root']))
    root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='where-', dir=root))
    for n in ['ASCEND_NPU_ARCH', 'TVM_HOME', 'TVM_IMPORT_PYTHON_PATH']:
        os.environ.pop(n, None)
    os.environ.update(TL_PLATFORM='A5', TL_ROOT=str(V), TILELANG_LIBRARY_PATH=str(C/'handoff/lib'),
                      TVM_LIBRARY_PATH=str(C/'handoff/lib'), SETUPTOOLS_USE_DISTUTILS='local', PYTHONDONTWRITEBYTECODE='1')
    for n in ['TILELANG_CACHE_DIR', 'TMPDIR', 'XDG_CACHE_HOME', 'TLA_CYTHON_CACHE_DIR', 'ASCEND_WORK_PATH',
              'ASCEND_CACHE_PATH', 'ASCEND_PROCESS_LOG_PATH', 'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR']:
        d = out/n
        d.mkdir(exist_ok=True)
        os.environ[n] = str(d)
    tempfile.tempdir = os.environ['TMPDIR']
    sys.dont_write_bytecode = True
    os.environ['PATH'] = str(C/'handoff/bin') + ':' + os.environ['PATH']
    sys.path[:0] = [str(R), str(V/'3rdparty/tvm/python'), str(C/'python'),
                   str(W/'.cache/all-pairs/remote_native_tla/python-deps'),
                   str(W/'TileOPs'), str(W/'TileOPs/src'), str(W)]
    sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
    import setuptools
    # Namespace the unchanged private helper; never replace public native.runtime.
    name = __package__ + '._integration700_tla_simple_tilelang_gemm7_private'
    package = types.ModuleType(name)
    package.__path__ = [str(P/'private_tla')]
    sys.modules[name] = package
    deployment = _module(name + '.deployment', P/'private_tla/deployment.py')
    native = _module(name + '.native', P/'private_tla/native.py')
    deployment.install_cache_hook(R, Path(os.environ['TLA_CYTHON_CACHE_DIR']))
    import tilelang
    if Path(tilelang.__file__).resolve() != (R/'tilelang/__init__.py').resolve():
        raise RuntimeError('Wrong TileLang runtime')
    if Path(tilelang._LIB_PATH).resolve() != (C/'handoff/lib/libtilelang_module.so').resolve():
        raise RuntimeError('Wrong TileLang library')
    native.VENDOR = V
    native.runtime = lambda: tilelang
    adapter = _module(name + '.adapter', P/'adapter.py')
    _CONTEXT = (native, tilelang, adapter, out)
    return _CONTEXT


def factory(op, case):
    if not matches(op,case): raise ValueError("Outside exact verified scope")
    if json.loads(json.dumps(case.params)) != config()['params'][op+'/'+case.id]: raise ValueError("Manifest changed")
    native, tilelang, adapter, out = _prepare()
    return adapter.factory(case, op, native, tilelang, out)
