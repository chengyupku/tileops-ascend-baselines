import sys,pathlib,os,json,time
P=pathlib.Path('/opt/torch-baseline1135-20261103');sys.path[:0]=[str(P/'source/src'),str(P/'source'),str(P),'/opt/tileops-ci/runtime/lib/python3.12/site-packages'];os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import torch,torch_npu
from benchmarks.api import cases
from workloads.device import set_run_device
from workloads.quantization.quantize import _pad_to_blocks
T=json.loads((P/'server2/runs/6a44e474a57aeff661e9/a2/task.json').read_text());torch.set_num_threads(8);set_run_device('cpu');torch.manual_seed(T['seed']);c=next(c for c in cases(T['key'][1]) if c.id==T['key'][2]);x=c.inputs[0];expected=c.reference(x);y=x.npu();torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype','ACL_OP_SELECT_IMPL_MODE':'high_precision'});got=c.reference(y);torch.npu.synchronize();got=tuple(t.cpu() for t in got);result={}
for label,a,b in zip(['codes','scales'],got,expected):
 av=a.view(torch.uint8).flatten();bv=b.view(torch.uint8).flatten();inds=(av!=bv).nonzero().flatten();result[label]={'mismatch_bytes':len(inds),'total_bytes':av.numel(),'indices':inds[:12].tolist(),'npu_bytes':av[inds[:12]].tolist(),'cpu_bytes':bv[inds[:12]].tolist()}
 if label=='scales':
  idx=(a!=b).flatten().nonzero().flatten();result[label]['samples']=[{'index':i,'cpu':b.flatten()[i].item(),'npu':a.flatten()[i].item()} for i in idx[:12].tolist()]
w=x.float();pad=_pad_to_blocks(w,True);mx=pad.view(pad.shape[0]//128,128,pad.shape[1]//128,128).abs().amax((1,3));high=(mx.double()/448).float();result['scales_cpu_vs_double']=torch.equal(expected[1].view(torch.uint8),high.view(torch.uint8));result['scales_npu_vs_double']=torch.equal(got[1].view(torch.uint8),high.view(torch.uint8));result['at']=time.time();(P/'server2/FP8_HIGH_PRECISION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));torch.npu.synchronize()
