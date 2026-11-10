import torch,torch_npu,json
from pathlib import Path
from tileops.kernels.ascend.elementwise import clamp_tensor
from tileops.ops.elementwise import ClampTensorFwdOp
values=[0.,-0.,-1.,1.,float('inf'),-float('inf'),float('nan')]
rows=[]
for dtype in [torch.float16,torch.float32]:
 x,lo,hi=[torch.tensor(values,dtype=dtype).reshape(s) for s in [(7,1,1),(1,7,1),(1,1,7)]]
 for mode in ['both','min','max']:
  cpu=(x,lo if mode!='max' else None,hi if mode!='min' else None)
  ref=torch.clamp(*cpu);pred=x
  if mode!='max':pred=torch.where(torch.isnan(lo)|(pred<=lo),lo,pred)
  if mode!='min':pred=torch.where(torch.isnan(hi)|(pred>=hi),hi,pred)
  op=ClampTensorFwdOp();out=op(*(a.npu() if a is not None else None for a in cpu)).cpu()
  z=(ref==0)&(pred==0);cpu_mismatch=int(((torch.signbit(ref)!=torch.signbit(pred))&z).sum())
  z=(ref==0)&(out==0);mask=(torch.signbit(ref)!=torch.signbit(out))&z
  indices=mask.nonzero().tolist();arrays=torch.broadcast_tensors(x,lo,hi)
  examples=[{'index':i,'x':float(arrays[0][tuple(i)]),'lo':float(arrays[1][tuple(i)]),'hi':float(arrays[2][tuple(i)]),'ref':float(ref[tuple(i)]),'out':float(out[tuple(i)])} for i in indices[:12]]
  row={'dtype':str(dtype),'mode':mode,'cpu_inclusive_mismatch':cpu_mismatch,'kernel_mismatch':int(mask.sum()),'examples':examples};rows.append(row);print(row,flush=True)
  k=[k for e in op._built_entries.values() for k in e.values()][0]
  Path(__file__).with_name('clamp-tensor-broadcast-'+str(dtype)+'-'+mode+'.c').write_text(k.compiled.get_kernel_source())
torch.npu.synchronize();Path(__file__).with_name('clamp-tensor-broadcast-probe.json').write_text(json.dumps(rows,indent=2)+'\n')
