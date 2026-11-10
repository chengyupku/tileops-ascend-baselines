import os,sys,pathlib,json,hashlib,importlib.abc
P=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(P/'source/src'),str(P/'source'),'/root/dyq/workspace/.venv/lib/python3.12/site-packages']
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU enumeration forbids NPU',name=fullname)
sys.meta_path.insert(0,NoNPU())
from benchmarks.api import cases
from tileops.manifest import load_manifest
rows=[];errors=[]
def norm(x):
 if isinstance(x,dict):return {str(k):norm(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [norm(v) for v in x]
 if isinstance(x,(str,int,float,bool)) or x is None:return x
 return str(x)
for op in load_manifest():
 try:
  for c in cases(op):
   key=['torch-eager',op,c.id];d={'op':op,'case_id':c.id,'tensors':norm(c._call.tensors),'params':norm(c.params)}
   tid=hashlib.sha256(json.dumps([key,d],sort_keys=True).encode()).hexdigest()[:20]
   refname='not-instantiated-cpu-catalog';err=None
   rows.append({'id':tid,'key':key,'case':d,'seed':0,'reference':refname,'reference_error':err})
 except Exception as e:errors.append({'op':op,'error':repr(e)})
(P/'CURRENT_CASES.json').write_text(json.dumps({'rows':rows,'errors':errors},indent=2))
print(json.dumps({'operators':len(set(r['key'][1] for r in rows)),'cases':len(rows),'missing_reference':[r['key'] for r in rows if not r['reference']],'errors':errors}))
