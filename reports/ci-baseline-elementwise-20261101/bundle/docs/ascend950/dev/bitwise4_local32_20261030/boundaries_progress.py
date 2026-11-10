"""Exact CPU oracle checks across dtype, tail, broadcast and noncontiguous inputs."""
import argparse
import importlib
import json
from pathlib import Path
import torch
import torch_npu
p=argparse.ArgumentParser();p.add_argument('--op',required=True);p.add_argument('--output',required=True);a=p.parse_args()
importlib.import_module('tileops.kernels.ascend.elementwise.'+{'BitwiseAndFwdOp':'bitwise_binary','BitwiseOrFwdOp':'bitwise_or','BitwiseXorFwdOp':'bitwise_xor','BitwiseNotFwdOp':'bitwise_not'}[a.op])
from tileops.ops import elementwise
Op=getattr(elementwise,a.op)
fn=getattr(torch,{'BitwiseAndFwdOp':'bitwise_and','BitwiseOrFwdOp':'bitwise_or','BitwiseXorFwdOp':'bitwise_xor','BitwiseNotFwdOp':'bitwise_not'}[a.op])
torch.npu.set_device(0)
rows=[]
for dtype in [torch.bool,torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64]:
 for shape,bshape in [((0,3),(1,3)),((),()),((1,),(1,)),((7,),(1,)),((129,),(129,)),((3,7),(7,)),((2,1,5),(1,3,1)),((3,5),(3,5)),((16387,),(16387,))]:
  x=torch.randint(0,127,shape,dtype=torch.int64).to(dtype)
  y=torch.randint(0,127,bshape,dtype=torch.int64).to(dtype)
  if dtype not in (torch.bool,torch.uint8) and x.numel():
   vals=torch.tensor([torch.iinfo(dtype).min,torch.iinfo(dtype).max,-1,0,1],dtype=dtype)
   x.reshape(-1)[:min(5,x.numel())]=vals[:min(5,x.numel())]
  if x.ndim==2 and x.numel():x=x.t().contiguous().t()
  inputs=[x] if a.op=='BitwiseNotFwdOp' else [x,y]
  refs=[z.clone() for z in inputs];dev=[z.npu() for z in inputs]
  result=Op()(*dev);torch.npu.synchronize()
  torch.testing.assert_close(result.cpu(),fn(*inputs),rtol=0,atol=0)
  for z,r in zip(dev,refs):torch.testing.assert_close(z.cpu(),r,rtol=0,atol=0)
  if result.numel():assert all(result.data_ptr()!=z.data_ptr() for z in dev)
  rows.append({'dtype':str(dtype),'shape':shape,'other_shape':bshape,'passed':True})
  Path(a.output).with_suffix('.progress.json').write_text(json.dumps({'op':a.op,'passed':len(rows),'last_dtype':str(dtype),'last_shape':shape}))
  print(a.op, str(dtype), shape, 'passed', flush=True)
Path(a.output).write_text(json.dumps({'op':a.op,'checks':rows,'passed':len(rows)},indent=2))
print(a.op,len(rows),'passed',flush=True)
