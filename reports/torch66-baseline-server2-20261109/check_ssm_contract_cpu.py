import os,sys,pathlib,json,importlib.abc
P=pathlib.Path(__file__).resolve().parent
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0';os.environ['FLA_NPU_DISABLE_PTH']='1'
sys.path[:0]=[str(P/'source/src'),str(P/'source'),str(P),'/opt/tileops-ci/runtime/lib/python3.12/site-packages']
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU only',name=fullname)
sys.meta_path.insert(0,NoNPU())
from benchmarks.api import cases
from tools.measure_baselines import _jsonable
for i in range(31,42):
 t=json.load(open(P/'tasks'/f'{i:02d}.json'));c=next(c for c in cases(t['key'][1]) if c.id==t['key'][2]);print(i,_jsonable(c.params)==_jsonable(t['case']['params']),flush=True)
