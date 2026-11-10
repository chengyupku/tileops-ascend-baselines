import pathlib,sys,os,importlib.abc,json,dataclasses,time
p=pathlib.Path(__file__).resolve().parent
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0';os.environ['FLA_NPU_DISABLE_PTH']='1'
sys.path[:0]=[str(p/'source/src'),str(p/'source'),str(p),'/opt/tileops-ci/runtime/lib/python3.12/site-packages']
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError(fullname,name=fullname)
sys.meta_path.insert(0,NoNPU())
import torch
from benchmarks.api import cases
from workloads.numerics import compare_outputs
from workloads.device import set_run_device
set_run_device('cpu');torch.set_num_threads(8);rows=[]
for task in json.loads((p/'server1/PLAN.json').read_text())['rows']:
 if task['key'][1]!='GQAPrefillPagedWithKVCacheFwdOp':continue
 torch.manual_seed(task['seed']);case=next(c for c in cases(task['key'][1]) if c.id==task['key'][2]);xs=case.inputs;out=case.reference(*xs)
 try:compare_outputs(out,out,case.verification);self_check={'passed':True}
 except Exception as e:self_check={'passed':False,'error':str(e)}
 rows.append({'id':task['id'],'key':task['key'],'seed':task['seed'],'CPU_reference_vs_itself':self_check,'output_nan_counts':[int(torch.isnan(x.float()).sum()) for x in out],'new_v_absmax':float(xs[2].float().abs().max()),'v_scale':float(xs[6][0]),'fp8_limit':448,'cpu_only':True})
 del xs,out,case
assert 'torch_npu' not in sys.modules
(p/'server1/GQA_CPU_SELF_CHECK.json').write_text(json.dumps({'at':time.time(),'rows':rows},indent=2));print(json.dumps(rows))
