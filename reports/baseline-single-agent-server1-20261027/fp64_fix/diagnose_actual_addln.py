import os,sys,importlib.abc
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0';os.environ['CUDA_VISIBLE_DEVICES']=''
class BlockDevice(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch_npu','acl','aclnn','tilelang','triton'):raise ImportError('CPU diagnostic forbids '+fullname)
sys.meta_path.insert(0,BlockDevice())
from pathlib import Path
import json,hashlib
import torch
import torch.nn.functional as F
torch.set_num_threads(4)
p=Path(__file__).resolve().parent
f=p/'runs_addln_original/ops-nn/llama-70b-prefill-float16/ORIGINAL_ADDLN_OUTPUT.pt'
d=torch.load(f,map_location='cpu',weights_only=True);x,r,w,b=d['inputs'];out,residual=d['outputs'];summed=x.float()+r.float()
expected_original=F.layer_norm(summed.to(x.dtype).float(),(x.shape[-1],),w.float(),b.float(),d['eps']).to(x.dtype)
expected_fused=F.layer_norm(summed,(x.shape[-1],),w.float(),b.float(),d['eps']).to(x.dtype)
def compare(a,b):
 return {'elements':a.numel(),'mismatched_at_original_tolerance':int((~torch.isclose(a,b,atol=.001,rtol=.001,equal_nan=True)).sum()),'max_abs_err':float((a.float()-b.float()).abs().max())}
report={'case':'llama-70b-prefill-float16','actual_original_api_output':str(f),'original_contract':compare(out,expected_original),'fp32_fused_sum_diagnostic':compare(out,expected_fused),'residual_output':compare(residual,summed.to(x.dtype)),'tolerance_unchanged':{'atol':.001,'rtol':.001},'official_acceptance_changed':False,'note':'Full actual tensors compared on CPU. Alternative semantic contract is diagnostic only; user decision pending.'}
(p/'ADD_LAYERNORM_ACTUAL_DIAGNOSIS.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
