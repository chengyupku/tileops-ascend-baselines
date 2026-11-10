import os,sys,pathlib,json,importlib.abc,gc,dataclasses
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
P=pathlib.Path(__file__).resolve().parent;B=pathlib.Path('/opt/torch-all2208-20261105')
sys.path[:0]=[str(P),str(B/'source/src'),str(B/'source'),str(B),'/opt/tileops-ci/runtime/lib/python3.12/site-packages']
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU validation forbids '+fullname,name=fullname)
sys.meta_path.insert(0,NoNPU())
import torch,adapters
from benchmarks.api import cases
from workloads.device import set_run_device
from workloads.numerics import compare_outputs
set_run_device('cpu');torch.set_num_threads(8)
checks=[]
for r in json.loads((P/'TASKS.json').read_text()):
 if r['op'] not in ('INT8QuantPerBlockFwdOp','FP8QuantPerBlockFwdOp','INT8DequantPerBlockFwdOp','INT4QuantPerGroupFwdOp'):continue
 out=P/'runs'/str(r['index']);saved=torch.load(out/'payload.pt',weights_only=False)
 c=next(c for c in cases(r['op']) if c.id==r['case']);c.__dict__['inputs']=saved['inputs']
 fn=adapters.factory(r['op'],c._call.params,c.reference);got=fn(*saved['inputs']);result=compare_outputs(got,saved['expected'],c.verification)
 checks.append({'index':r['index'],'op':r['op'],'case':r['case'],'check':dataclasses.asdict(result)})
 (P/'CPU_FIXED_CHECKS.json').write_text(json.dumps(checks,indent=2,default=str))
 print('CPU_PASS',r['index'],flush=True);del saved,got,c,fn;gc.collect()
