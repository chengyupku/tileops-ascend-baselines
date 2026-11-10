import pathlib,sys,os,json
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
P=pathlib.Path(__file__).resolve().parent;sys.path[:0]=[str(P/'source/src'),str(P/'source'),str(P)]
import torch
from types import SimpleNamespace as NS
from workloads.device import set_run_device
set_run_device('cpu');torch.set_num_threads(2)
from adapters.backward_only_v2 import bind_gqa,bind_gla,bind_engram
from workloads.attention.gqa.bwd import GQABwdWorkload
from workloads.linear_attention.gla import GLAChunkwiseCall,gla_fwd_chunked_torch
from workloads.sequence_modeling.engram import ref_engram_gate_conv_bwd
results=[]
w=GQABwdWorkload(1,2,1,8,4,True,torch.float32);a=w.gen_inputs();expected=w.ref_program(*a);b=bind_gqa(w,a)
for _ in range(3):assert all(torch.allclose(x,y,atol=1e-6,rtol=1e-6) for x,y in zip(expected,b.execute()))
assert all(torch.count_nonzero(t)==0 for t in b.execute(torch.zeros_like(a[0])))
results.append('gqa')
for initial_grad in (False,True):
 q,k,v=[torch.randn(1,8,2,4)*.1 for _ in range(3)];g=-torch.rand_like(q)*.1;init=torch.randn(1,2,4,4)*.1 if initial_grad else None
 _,h=gla_fwd_chunked_torch(q,k,v,g,4,scale=.5,initial_state=init,with_chunk_states=True)
 a=(torch.randn_like(v),torch.randn_like(h[:,-1]),q,k,v,g,h)
 w=NS(arguments=lambda:{'chunk_size':4,'scale':.5,'has_initial_state':initial_grad},call=NS(signature=NS(name='GLAChunkBwdOp')))
 expected=GLAChunkwiseCall.ref_program(w,*a);b=bind_gla(w,a)
 for _ in range(3):assert all((x is None and y is None) or torch.allclose(x,y,atol=1e-6,rtol=1e-5) for x,y in zip(expected,b.execute()))
 assert all(t is None or torch.count_nonzero(t)==0 for t in b.execute(tuple(torch.zeros_like(t) for t in b.args[0])))
 results.append('gla_initial_'+str(initial_grad))
H,k,v=[torch.randn(1,8,4)*.1 for _ in range(3)];a=(torch.randn_like(H),H,k,v,torch.rand(4),torch.rand(4),torch.rand(3,4),None,None,None,None,None)
expected=ref_engram_gate_conv_bwd(*a);b=bind_engram(NS(eps=1e-6),a)
for _ in range(3):assert all(torch.allclose(x,y,atol=1e-6,rtol=1e-5) for x,y in zip(expected,b.execute()))
assert all(torch.count_nonzero(t)==0 for t in b.execute(torch.zeros_like(a[0])))
results.append('engram');(P/'BACKWARD_V2_CPU_TEST.json').write_text(json.dumps({'passed':results,'repeats_each':3,'device_validation':'pending'},indent=2));print(results)
