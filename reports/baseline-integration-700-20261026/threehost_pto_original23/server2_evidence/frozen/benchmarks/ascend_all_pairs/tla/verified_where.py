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
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = json.loads((HERE / 'verified_where.json').read_text())
    return _CONFIG


def matches(op, case):
    return ["tilelang-ascend", op, case.id] == config()['key']


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
    name = __package__ + '._verified_where_private'
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


def factory(case):
    if not matches('WhereFwdOp', case):
        raise ValueError('Outside verified Where exact scope')
    # A matching ID cannot silently credit a changed manifest contract.
    if json.loads(json.dumps(case.params)) != config()['params']:
        raise ValueError('Verified Where manifest parameters changed')
    native, tilelang, adapter, out = _prepare()
    return adapter.factory(case, 'WhereFwdOp', native, tilelang, out)
