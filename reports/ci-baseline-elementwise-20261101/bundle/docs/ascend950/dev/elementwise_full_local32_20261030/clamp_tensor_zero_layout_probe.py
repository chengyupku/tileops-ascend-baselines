import torch,json
from pathlib import Path
rows=[]
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 for n in [1,7,31,32,33,63,64,65,127,128,129,343]:
  x=torch.zeros(n,dtype=dtype);lo=torch.full((n,),-0.,dtype=dtype)
  for mode in ['min','max','both']:
   y=torch.clamp(x,min=lo if mode!='max' else None,max=lo if mode!='min' else None)
   row={'dtype':str(dtype),'n':n,'mode':mode,'negative_zeros':int(torch.signbit(y).sum())};rows.append(row);print(row,flush=True)
Path(__file__).with_name('clamp-tensor-zero-layout-probe.json').write_text(json.dumps(rows,indent=2)+'\n')
