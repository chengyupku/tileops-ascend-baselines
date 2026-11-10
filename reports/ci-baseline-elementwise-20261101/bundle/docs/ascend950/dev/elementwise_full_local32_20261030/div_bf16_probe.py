"""Bitwise BF16 division diagnosis across the original public layouts."""
import json,hashlib
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import arithmetic_div as module
from tileops.ops.elementwise import DivFwdOp
from div_reference import div_reference
rows=[]
for xs,bs in [((65536,),(65536,)),((64,1024),(1,1024))]:
 gen=torch.Generator(device='npu').manual_seed(0)
 a=torch.randint(-32768,32768,xs,generator=gen,device='npu').to(torch.int16).view(torch.bfloat16)
 b=torch.randint(-32768,32768,bs,generator=gen,device='npu').to(torch.int16).view(torch.bfloat16)
 y=DivFwdOp()(a,b).cpu();x=a.cpu();z=b.cpu();ref=div_reference(x,z)
 bad=(y.view(torch.int16)!=ref.view(torch.int16))&~(y.isnan()&ref.isnan())
 xx,zz=torch.broadcast_tensors(x,z);ids=bad.flatten().nonzero().flatten();special=ref.abs()<torch.finfo(torch.bfloat16).tiny
 examples=[]
 for i in ids[:24].tolist():examples.append({'index':i,'a':xx.flatten()[i].item(),'b':zz.flatten()[i].item(),'actual':y.flatten()[i].item(),'expected':ref.flatten()[i].item()})
 row={'a_shape':xs,'b_shape':bs,'mismatches':int(bad.sum()),'tiny_output_mismatches':int((bad&special).sum()),'examples':examples};rows.append(row);print(row,flush=True)
torch.npu.synchronize()
Path(__file__).with_name('div-bf16-probe.json').write_text(json.dumps({'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
