from pathlib import Path
import sys,json,importlib.machinery,platform,ast,hashlib
P=Path(__file__).resolve().parent;W=P.parents[2];C=W/'.cache/all-pairs/remote_tla_compiler'
paths=[str(C/'python'),str(W/'.cache/all-pairs/remote_native_tla/python-deps'),*sys.path,'/opt/tileops-ci/runtime/lib/python3.12/site-packages']
r=dict(python=sys.version,stdlib={},dependencies={},syntax={})
for n in ['ctypes','sysconfig','typing','tempfile','setuptools','torch','torch_npu','numpy','Cython','tvm_ffi']:
 spec=importlib.machinery.PathFinder.find_spec(n,paths);r['dependencies'][n]=spec.origin if spec else None
for f in P.glob('*.py'):
 ast.parse(f.read_text());r['syntax'][f.name]='ok'
assert all(r['dependencies'][n] for n in ['setuptools','torch','torch_npu','numpy','Cython'])
(P/'CPU_CHECK.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
