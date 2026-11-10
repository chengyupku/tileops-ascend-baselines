import pathlib,json,hashlib,fcntl,difflib,datetime
P=pathlib.Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/tilelang_register5');D=pathlib.Path('/opt/tileops-all-pairs/workspace/TileOPs/benchmarks/ascend_all_pairs/tilelang')
module='integration700_tilelang_register5';src=(P/(module+'.py')).read_bytes();compile(src,str(P/(module+'.py')),'exec')
append='''
# integration700_tilelang_register5: exactly five previously checked original TileLang FP8 cases.
from .integration700_tilelang_register5 import factory as _register5_factory, KEYS as _register5_keys
_register5_previous = {op: PROVIDERS.get(op) for op, _ in _register5_keys}
def _register5_dispatch(case):
    op = case._call.signature.name
    if (op, case.id) in _register5_keys:
        return _register5_factory(case)
    previous = _register5_previous[op]
    if previous is None:
        raise NotImplementedError((op, case.id))
    return previous(case)
PROVIDERS.update({op: _register5_dispatch for op, _ in _register5_keys})
'''
with open('/tmp/baseline700-provider-tilelang.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 path=D/'__init__.py';old=path.read_bytes();assert module.encode() not in old;assert not (D/(module+'.py')).exists()
 (P/'attempt02/provider_init.before.py').write_bytes(old)
 new=old+append.encode();compile(new,str(path),'exec')
 (D/(module+'.py')).write_bytes(src);path.write_bytes(new)
 (P/'attempt02/provider_init.after.py').write_bytes(new)
 (P/'attempt02/registration.patch').write_text(''.join(difflib.unified_diff(old.decode().splitlines(True),new.decode().splitlines(True),fromfile=str(path),tofile=str(path))))
 (P/'attempt02/REGISTRATION.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),lock='/tmp/baseline700-provider-tilelang.lock',before_sha256=hashlib.sha256(old).hexdigest(),after_sha256=hashlib.sha256(new).hexdigest(),module_sha256=hashlib.sha256(src).hexdigest(),preserved_previous_bytes=new.startswith(old)),indent=2))
print('registered five exact keys')
