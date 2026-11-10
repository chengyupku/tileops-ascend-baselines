import torch,json,struct
from pathlib import Path
rows=[]
bounds=[(None,0.),(0.,None),(-0.,None),(None,-0.),(-0.,0.),(0.,-0.),(-1.,1.),(1.,-1.),(float('nan'),1.),(-1.,float('nan')),(None,float('nan')),(float('nan'),None),(float('inf'),None),(None,-float('inf'))]
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 for n in [7,65,128]:
  seed=torch.tensor([0.,-0.,-1.,1.,float('inf'),-float('inf'),float('nan')],dtype=dtype);x=seed.repeat((n+6)//7)[:n]
  for lo,hi in bounds:
   ref=torch.clamp(x,min=lo,max=hi);counts={}
   for inclusive in [False,True]:
    y=x
    if lo is not None:
     b=torch.full_like(x,lo);y=torch.where(torch.isnan(b)|((y<=b) if inclusive else (y<b)),b,y)
    if hi is not None:
     b=torch.full_like(x,hi);y=torch.where(torch.isnan(b)|((y>=b) if inclusive else (y>b)),b,y)
    equal=(y==ref)|(torch.isnan(y)&torch.isnan(ref));z=(y==0)&(ref==0)
    counts['inclusive' if inclusive else 'strict']=int(((~equal)|(z&(torch.signbit(y)!=torch.signbit(ref)))).sum())
   row={'dtype':str(dtype),'n':n,'min':None if lo is None else struct.pack('!d',lo).hex(),'max':None if hi is None else struct.pack('!d',hi).hex(),'mismatches':counts};rows.append(row)
print(json.dumps(rows,indent=2),flush=True)
Path(__file__).with_name('clamp-scalar-cpu-probe.json').write_text(json.dumps(rows,indent=2)+'\n')
