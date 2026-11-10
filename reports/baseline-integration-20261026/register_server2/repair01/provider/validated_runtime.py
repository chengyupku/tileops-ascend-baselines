"""CPU preflight and hash-pinned private import relocation; no runtime mutation."""
import hashlib
import json
import os
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / 'validated_manifest.json').read_text())
NAMESPACE = '_ops_transformer_verified_20261026'

def verify_pins(group):
    cfg = MANIFEST['groups'][group]
    for directory in cfg['directories']:
        if not Path(directory).is_dir():
            raise RuntimeError('Required directory missing: ' + directory)
    for filename, expected in cfg['pins'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Pinned source/runtime changed: ' + filename)

def require_guard(group):
    meta = json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
    owner = 'selected-integration-20261026/transformer_' + group
    if (meta.get('owner') != owner or meta.get('phase') != 'running'
            or meta.get('child', {}).get('pid') != os.getpgrp()):
        raise RuntimeError('Matching foreground npu-guard reservation required')

def check_environment(group):
    cfg = MANIFEST['groups'][group]
    startup = dict(item.split('=', 1) for item in Path('/proc/self/environ').read_bytes().decode().split('\0') if '=' in item)
    for key, value in cfg['environment'].items():
        if os.environ.get(key) != value or startup.get(key) != value:
            raise RuntimeError('Pre-exec environment mismatch: ' + key)
    prefix = cfg.get('ld_library_path_prefix')
    if prefix and (not os.environ.get('LD_LIBRARY_PATH', '').startswith(prefix)
            or startup.get('LD_LIBRARY_PATH') != os.environ.get('LD_LIBRARY_PATH')):
        raise RuntimeError('Pre-exec attention LD_LIBRARY_PATH mismatch')
    for path in cfg['python_paths']:
        if path not in sys.path:
            raise RuntimeError('Missing Python search path: ' + path)
    run = Path(os.environ['TRANSFORMER_RUN_DIR']).resolve()
    allowed = Path(MANIFEST['run_root']).resolve()
    if not run.is_relative_to(allowed) or not run.is_dir():
        raise RuntimeError('Run directory outside provider-owned report root')
    if Path.cwd() != (Path(cfg['cwd']) if group == 'gdn' else run):
        raise RuntimeError('Working directory mismatch')
    if group == 'gdn' and os.environ.get('GDN_RUN_DIR') != str(run):
        raise RuntimeError('GDN_RUN_DIR mismatch')
    return run

def fallback(case):
    from benchmarks.ops.bench_ascend_classics import candidate_factories
    from benchmarks.ascend_baselines import UnsupportedCase
    factories = candidate_factories(case)
    if 'ops-transformer' not in factories:
        raise UnsupportedCase('No historical ops-transformer factory for this case')
    return factories['ops-transformer']()

def contract(case, group):
    c = MANIFEST['contracts'].get(case.id)
    if c is None or c['group'] != group:
        return False
    snap = c['snapshot']
    if case._call.signature.name != c['op'] or json.loads(json.dumps(case.params)) != snap['parameters']:
        raise ValueError('Verified case contract changed: ' + case.id)
    # Validate actual supplied tensors as well as declared parameters, without importing torch.
    for name, tensor in zip(snap['declared_input_names'], case.inputs):
        declared = snap['declared_tensors'].get(name)
        if declared is not None and (tensor is None or list(tensor.shape) != declared[0]
                or str(tensor.dtype).removeprefix('torch.') != declared[1]):
            raise ValueError('Verified input contract changed: ' + name)
    return True

def relocated_source(name):
    filename = MANIFEST['sources'][name]
    group = 'gdn' if name in ('binding', 'direct_acl', 'resource_gate') else 'attention'
    data = Path(filename).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != MANIFEST['groups'][group]['pins'][filename]:
        raise RuntimeError('Private source pin mismatch: ' + filename)
    source = data.decode()
    replacements = {
        'from direct_acl import': f'from {NAMESPACE}.direct_acl import',
        'from benchmarks.ascend_graph_call import': f'from {NAMESPACE}.ascend_graph_call import',
        'from benchmarks.ascend_transform_baselines import': f'from {NAMESPACE}.ascend_transform_baselines import',
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    return filename, source, digest

def load(name):
    filename, source, digest = relocated_source(name)
    if NAMESPACE not in sys.modules:
        package = types.ModuleType(NAMESPACE)
        package.__path__ = []
        sys.modules[NAMESPACE] = package
    key = NAMESPACE + '.' + name
    if key in sys.modules:
        module = sys.modules[key]
        if getattr(module, '_verified_digest', None) != digest or module.__file__ != filename:
            raise RuntimeError('Private module cache conflict: ' + key)
        return module
    dependencies = {'binding': ['direct_acl'], 'ascend_transformer_graph': ['ascend_graph_call', 'ascend_transform_baselines']}
    for dependency in dependencies.get(name, []):
        load(dependency)
    module = types.ModuleType(key)
    module.__file__ = filename  # Preserve original HERE-relative recipes and source attribution.
    module.__package__ = NAMESPACE
    module._verified_digest = digest
    sys.modules[key] = module
    try:
        exec(compile(source, filename, 'exec'), module.__dict__)
    except BaseException:
        del sys.modules[key]
        raise
    return module

def preflight(case, group):
    verify_pins(group)
    run = check_environment(group)
    require_guard(group)
    if group == 'gdn':
        if run.name != case.id:
            raise RuntimeError('Resource gate requires exact case run directory')
        chunk = Path(MANIFEST['groups'][group]['root']) / 'chunk_package'
        load('resource_gate').require_capacity(chunk, run)
    return run
