import sys,importlib.abc,json,hashlib
from pathlib import Path
from types import SimpleNamespace
P=Path(__file__).resolve().parent;W=P.parents[2]
sys.path.insert(0,str(W/'TileOPs'))
class NoDevice(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','triton'):raise RuntimeError('Device import prohibited: '+fullname)
sys.meta_path.insert(0,NoDevice())
from benchmarks.ascend_all_pairs import ops_transformer,ops_tensor
checked=[];hashes={}
for module,file in [(ops_transformer,'REGISTRATION.json'),(ops_tensor,'REGISTRATION_GROUPED.json')]:
 reg=json.load(open(P/file))
 for name,digest in reg['modules'].items():
  actual=Path(module.__file__).parent/name
  assert hashlib.sha256(actual.read_bytes()).hexdigest()==digest
  assert actual.read_bytes()==(P/name).read_bytes()
  hashes[str(actual)]=digest
 for op,cid in reg['keys']:
  tag='_i700_misc_'+op if module is ops_transformer else '_i700_misc_grouped'
  old=getattr(module,tag)
  setattr(module,tag,lambda case:('selected',case.id))
  try:assert module.PROVIDERS[op](SimpleNamespace(id=cid))==('selected',cid)
  finally:setattr(module,tag,old)
  checked.append([module.__name__,op,cid])
d={'device_free':True,'blocked_imports':['torch','torch_npu','triton'],'checked_keys':checked,'deployed_matches_numerically_verified_private_modules':True,'source_hashes':hashes}
(P/'REGISTRATION_CHECK.json').write_text(json.dumps(d,indent=2));print(len(checked))
