import sys,json,types
from pathlib import Path
P=Path(__file__).resolve().parent;W=P.parents[2]
# Device-free namespace stubs avoid broad benchmark imports.
for name,path in [('benchmarks',W/'TileOPs/benchmarks'),('benchmarks.ascend_all_pairs',W/'TileOPs/benchmarks/ascend_all_pairs')]:
 m=types.ModuleType(name);m.__path__=[str(path)];sys.modules[name]=m
from benchmarks.ascend_all_pairs import tla
m=tla._i700_tla_simple_select;m.inspect_pins()
old=tla._i700_tla_simple_previous['WhereFwdOp']
case=types.SimpleNamespace(id='256M-bfloat16')
assert m.matches('WhereFwdOp',case)
assert not m.matches('WhereFwdOp',types.SimpleNamespace(id='256M-float16'))
assert tla.PROVIDERS['WhereFwdOp'].func==tla._i700_tla_simple_dispatch
assert not any(x in sys.modules for x in ['torch','torch_npu','tilelang','tvm'])
(P/'REGISTRY_CHECK.json').write_text(json.dumps(dict(pins_valid=True,exact_key_routing=True,legacy_where_fp16_preserved=True,device_imports=False),indent=2))
import importlib
records=[]
for f in P.glob('REGISTERED_*.json'):
 r=json.loads(f.read_text());module=importlib.import_module(r['factory'].split(':')[0]);module.inspect_pins()
 for op,cid in r['keys']:
  assert module.matches(op,types.SimpleNamespace(id=cid))
  assert not module.matches(op,types.SimpleNamespace(id='OUTSIDE_SCOPE'))
 records.append(dict(file=f.name,keys=r['keys'],pins_valid=True))
assert not any(x in sys.modules for x in ['torch','torch_npu','tilelang','tvm'])
(P/'GROUP_REGISTRY_CHECK.json').write_text(json.dumps(records,indent=2))
print('device-free registration checks passed')
