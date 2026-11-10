from pathlib import Path
import importlib.util,sys,json
from types import SimpleNamespace
P=Path(__file__).resolve().parent
before=set(sys.modules)
spec=importlib.util.spec_from_file_location('isnan_proposal',P/'lazy_provider_proposal.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert not any(n.split('.')[0] in ('torch','torch_npu','tilelang','tvm') for n in set(sys.modules)-before)
c=SimpleNamespace(id='16M-float32',params=m.PARAMS)
assert m.matches('IsnanFwdOp',c);m.validate_case('IsnanFwdOp',c)
for op,case in [('IsinfFwdOp',c),('IsnanFwdOp',SimpleNamespace(id='256M-float32',params=m.PARAMS)),('IsnanFwdOp',SimpleNamespace(id=c.id,params={'input':[[4096,4096],'float16'],'output':[[4096,4096],'bool']}))]:
 try:m.validate_case(op,case)
 except ValueError:pass
 else:raise AssertionError('Out of contract accepted')
called=[]
def prepare():
 called.append(True)
 return 'native','tl',SimpleNamespace(factory=lambda *args:args),'out'
m._prepare=prepare
result=m.factory('IsnanFwdOp',c)
assert called==[True] and result==(c,'IsnanFwdOp','native','tl','out')
(P/'STATIC_ROUTING.json').write_text(json.dumps(dict(passed=True,device_imports=False,positive_routes=1,negative_routes=3,factory_delegation=True,proposal_device_tested=False),indent=2))
print('CPU lazy import and exact route checks passed')
