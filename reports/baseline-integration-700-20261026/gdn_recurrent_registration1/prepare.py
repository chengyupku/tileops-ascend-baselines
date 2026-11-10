from pathlib import Path
import json,hashlib
O=Path(__file__).resolve().parent; P=O.parent/'gdn_capacity_recurrent1'; key='q35-gva-decode-b512-bfloat16'
h=lambda b:hashlib.sha256(b).hexdigest()
files=['recurrent_package/'+n for n in ['binding.py','direct_acl.py','entry.py','resource_gate.py','RUNTIME_RECIPE.json','RUNTIME_PINS.json','RESOURCE_ESTIMATES.json']]+['RECURRENT_CASES.json','CONTRACTS.json','CANDIDATE_CASES.json','CASE_RESOURCE_ESTIMATES.json',key+'/child.sh']
pins={n:h((P/n).read_bytes()) for n in files}
native=json.loads((P/'NATIVE_EVIDENCE.json').read_text()); runtime=dict(native['source_artifact_sha256']);runtime.update(native['loaded_original_opapi']);runtime.update({x['path']:x['sha256'] for x in json.loads((P/key/'loaded-libraries.json').read_text())})
recipe=json.loads((P/'recurrent_package/RUNTIME_RECIPE.json').read_text());params=json.loads((P/'CONTRACTS.json').read_text())[key]['params']
header='''"""Private frozen registration; no test switch and no numerical implementation."""
import builtins, hashlib, json, os, types
from pathlib import Path
from types import MappingProxyType
'''
header+=f'ROOT = Path({str(P)!r})\nCASE_ID = {key!r}\nPARAMS_JSON = {json.dumps(params,sort_keys=True,separators=(",",":"))!r}\nFILE_PINS = MappingProxyType({pins!r})\nRUNTIME_PINS = MappingProxyType({runtime!r})\nENV = MappingProxyType({recipe["required_environment"]!r})\nLD_PREFIX = {tuple(recipe["ld_library_path_order"])!r}\n'
body='''
def _checked(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError('Frozen GDN hash mismatch: ' + str(path))
    return raw

def _snapshot():
    return MappingProxyType({str(ROOT / n): _checked(ROOT / n, pin) for n, pin in FILE_PINS.items()})

class _FrozenPath:
    def __init__(self, path, snapshot):
        self.path, self.snapshot = path, snapshot
    @property
    def parent(self): return _FrozenPath(self.path.parent, self.snapshot)
    def __truediv__(self, name): return _FrozenPath(self.path / name, self.snapshot)
    def __str__(self): return str(self.path)
    def read_bytes(self): return self.snapshot[str(self.path)]
    def read_text(self): return self.read_bytes().decode()

def _execute(name, filename, raw, transport=None):
    mod = types.ModuleType(name)
    mod.__file__ = str(filename)
    imports = dict(vars(builtins))
    original = builtins.__import__
    def load(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'direct_acl' and level == 0 and transport is not None:
            return transport
        return original(name, globals, locals, fromlist, level)
    imports['__import__'] = load
    mod.__dict__['__builtins__'] = imports
    exec(compile(raw, str(filename), 'exec'), mod.__dict__)
    return mod

def _environment():
    for name, value in ENV.items():
        if name == 'GDN_CASE_OWNER':
            if not os.environ.get(name): raise RuntimeError('GDN_CASE_OWNER required')
        elif os.environ.get(name) != value:
            raise RuntimeError('Original GDN environment required: ' + name)
    if tuple(os.environ.get('LD_LIBRARY_PATH', '').split(':')[:len(LD_PREFIX)]) != LD_PREFIX:
        raise RuntimeError('Original GDN LD_LIBRARY_PATH order required')

def build(case):
    if case.id != CASE_ID or json.dumps(case.params,sort_keys=True,separators=(',',':'),allow_nan=False) != PARAMS_JSON:
        raise ValueError('Exact GDN case and all parameters required')
    snapshot = _snapshot()
    package = ROOT / 'recurrent_package'
    transport = _execute('_frozen_gdn_transport', package/'direct_acl.py', snapshot[str(package/'direct_acl.py')])
    transport.require_guard()
    _environment()
    for path, pin in RUNTIME_PINS.items():
        _checked(Path(path), pin)
    binding = _execute('_frozen_gdn_binding', package/'binding.py', snapshot[str(package/'binding.py')], transport)
    binding.HERE = _FrozenPath(package, snapshot)
    return binding.build(types.SimpleNamespace(id=CASE_ID, params=json.loads(PARAMS_JSON)))

def wrap(previous):
    def dispatch(case):
        if case.id == CASE_ID: return build(case)
        if previous is None: raise KeyError(case.id)
        return previous(case)
    return dispatch
'''
(O/'integration700_gdn_recurrent_registration1.py').write_text(header+body)
(O/'FROZEN_CONFIG_HASHES.json').write_text(json.dumps(pins,indent=2)+'\n')
(O/'EXACT_CONTRACT.json').write_text(json.dumps({'case_id':key,'params':params},indent=2)+'\n')
(O/'ENVIRONMENT.json').write_text(json.dumps({'recipe':recipe,'actual_successful_child':str(P/key/'child.sh'),'actual_child_sha256':pins[key+'/child.sh'],'requirements':['Use original child environment, including vendor set_env.bash before final custom OPP assignment; original python -S -B and bootstrap paths in entry.py.','Set GDN_CASE_OWNER to actual authorized foreground guard owner; require_guard checks running metadata and process group.','File availability and CPU proof do not establish runtime availability.','This worker authorizes no NPU runs or numerical reruns.']},indent=2)+'\n')
print('module ready',len(runtime),'precise runtime/source pins',len(pins),'configuration pins')
