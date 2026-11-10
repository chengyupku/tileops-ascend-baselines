import torch,json,struct
from pathlib import Path
rows=[]
parameters=[(0.,None,None),(-0.,0.,-0.),(.3,.7,-.3),(float('inf'),-float('inf'),float('nan')),(float('nan'),float('nan'),float('nan')),(1e40,1e40,-1e40),(1e-50,1e-50,-1e-50),(0.,float('inf'),-float('inf')),(1e6,1e6,-1e6)]
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 f=torch.finfo(dtype);x=torch.tensor([float('nan'),float('inf'),-float('inf'),0.,-0.,1.,-1.,f.tiny/2,-f.tiny/2,f.max,-f.max],dtype=dtype)
 for nan,pos,neg in parameters:
  y=torch.nan_to_num(x,nan=nan,posinf=pos,neginf=neg)
  assert torch.equal(y[3:],x[3:]) and torch.equal(torch.signbit(y[3:]),torch.signbit(x[3:]))
  row={'dtype':str(dtype),'parameters':[None if v is None else struct.pack('!d',v).hex() for v in [nan,pos,neg]],'replacements':y[:3].tolist(),'finite_unchanged':True};rows.append(row);print(row,flush=True)
Path(__file__).with_name('nan-to-num-cpu-probe.json').write_text(json.dumps(rows,indent=2)+'\n')
