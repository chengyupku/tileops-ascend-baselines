import json,runpy,types,copy,sys,importlib.abc,ctypes
from pathlib import Path
P=Path(__file__).resolve().parent
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ['torch','torch_npu','acl','tilelang','tvm']:raise AssertionError('Device dependency imported: '+name)
sys.meta_path.insert(0,Block())
def forbidden(*a,**kw):raise AssertionError('Native load')
ctypes.CDLL=forbidden;ctypes.PyDLL=forbidden
m=runpy.run_path(str(P/'registration.py'));rows=json.loads((P/'CONTRACTS.json').read_text());count=0
for r in rows:
 c=types.SimpleNamespace(id=r['case_id'],params=copy.deepcopy(r['parameters']),_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=r['op'],inputs=r['declared_input_names'],outputs=r['declared_output_names']),tensors=copy.deepcopy(r['declared_tensors'])))
 assert m['matches'](c);count+=1
 c.id+='-unscoped';assert not m['matches'](c);c.id=r['case_id'];count+=1
 c.params['unexpected_parameter']=True;assert not m['matches'](c);del c.params['unexpected_parameter'];count+=1
 c._call.tensors={};assert not m['matches'](c);count+=1
 sentinel=object();assert m['wrap'](lambda x:sentinel)(c) is sentinel;count+=1
(P/'CONTRACT_CHECKS.json').write_text(json.dumps({'checks':count,'exact_keys':len(rows),'all_passed':True,'device_imports':0,'native_loads':0,'shared_registry_writes':0},indent=2));print(count)
