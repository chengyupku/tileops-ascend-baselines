import sys,os,json,pathlib
P=pathlib.Path('/opt/torch-baseline1135-20261103');sys.path[:0]=[str(P/'source/src'),str(P/'source'),'/opt/tileops-ci/runtime/lib/python3.12/site-packages']
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import torch,torch_npu
from benchmarks.api import cases
from workloads.device import set_run_device
from workloads.quantization.quantize import _pad_to_blocks
set_run_device('cpu');torch.set_num_threads(8)
results=[]
for tid in ['58a126c2a6b8fa945dc5','cdd6c8593df965d79c76','ea7d135338f826c53bd3']:
 task=json.loads(next((P/'server1/runs'/tid).glob('a*/task.json')).read_text());torch.manual_seed(task['seed']);c=next(c for c in cases(task['key'][1]) if c.id==task['key'][2]);x=c.inputs[0];expected=c.reference(x);w=x.npu().float();n,k=w.shape;p=_pad_to_blocks(w,True);am=p.view(p.shape[0]//128,128,p.shape[1]//128,128).abs().amax((1,3));row={'id':tid,'modes':{}}
 for mode in ['scalar','tensor','reciprocal']:
  scale=am/448 if mode=='scalar' else (am/torch.full_like(am,448) if mode=='tensor' else am*torch.full_like(am,448).reciprocal());scale=torch.where(am>0,scale,torch.ones_like(scale));full=scale.repeat_interleave(128,0).repeat_interleave(128,1)[:n,:k];q=(w/full).clamp(-448,448).to(torch.float8_e4m3fn);torch.npu.synchronize();row['modes'][mode]={'scale_diff':int((scale.cpu().view(torch.int32)!=expected[1].view(torch.int32)).sum()),'code_diff':int((q.cpu().view(torch.uint8)!=expected[0].view(torch.uint8)).sum())}
 results.append(row);print(json.dumps(row),flush=True)
 del x,expected,w,p,am,scale,full,q,c
 torch.npu.empty_cache()
(P/'server1/fp8_div_probe_v1_result.json').write_text(json.dumps(results,indent=2));torch.npu.synchronize()
