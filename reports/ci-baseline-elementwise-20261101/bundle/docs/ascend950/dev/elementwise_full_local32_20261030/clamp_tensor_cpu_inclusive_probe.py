import torch,itertools,json
from pathlib import Path
rows=[]
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 values=[0.,-0.,-1.,1.,float('inf'),-float('inf'),float('nan')]
 x,lo,hi=[torch.tensor(v,dtype=dtype) for v in zip(*itertools.product(values,repeat=3))]
 for mode in ['both','min','max']:
  y=torch.clamp(x,min=lo if mode!='max' else None,max=hi if mode!='min' else None)
  pred=x.clone()
  if mode!='max':pred=torch.where(torch.isnan(lo)|(pred<=lo),lo,pred)
  if mode!='min':pred=torch.where(torch.isnan(hi)|(pred>=hi),hi,pred)
  eq=(y==pred)|(torch.isnan(y)&torch.isnan(pred));z=(y==0)&(pred==0)
  mismatch=(~eq)|(z&(torch.signbit(y)!=torch.signbit(pred)))
  indices=mismatch.nonzero().flatten().tolist()
  row={'dtype':str(dtype),'mode':mode,'inclusive_select_mismatches':len(indices),'examples':[{'x':float(x[i]),'lo':float(lo[i]),'hi':float(hi[i]),'ref':float(y[i]),'select':float(pred[i])} for i in indices[:15]]}
  rows.append(row);print(row,flush=True)
Path(__file__).with_name('clamp-tensor-cpu-probe-inclusive.json').write_text(json.dumps(rows,indent=2)+'\n')
