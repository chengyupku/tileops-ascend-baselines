"""Positive near-integer ratios that exercise the bounded fast branch in all layouts."""
import hashlib,json
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import arithmetic_modulo as module
from tileops.ops.elementwise import FloorDivideFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
from floor_divide_reference import floor_divide_reference
rows=[];torch.manual_seed(441)
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 limit={torch.float16:8192,torch.bfloat16:65536,torch.float32:4194304}[dtype]
 storage=torch.int16 if dtype.itemsize==2 else torch.int32
 for layout in ['same','bias','channel']:
  xs,bs={'same':((65536,),(65536,)),'bias':((64,960),(1,960)),'channel':((2,4,8192),(4,1))}[layout]
  b=(torch.rand(bs)*1.75+.25).to(dtype)
  k=torch.randint(1,limit//2,xs).double()
  center=(k*b.double()).to(dtype);raw=center.view(storage)
  op=FloorDivideFwdOp()
  for shift in [-1,0,1]:
   x=(raw+shift).view(dtype)
   ratio=x.double()/b.double()
   assert bool(((ratio>0)&(ratio<limit)).all())
   a=x.npu();c=b.npu();actual=op(a,c).cpu();expected=floor_divide_reference(x,b)
   compare_outputs(actual,expected,ElementwiseWorkload('FloorDivideFwdOp',(x,b)).verification(x,b))
   assert torch.equal(a.cpu().view(storage),x.view(storage)) and torch.equal(c.cpu(),b)
   row={'dtype':str(dtype),'layout':layout,'neighbor':shift,'values':x.numel(),'passed':True,'input_unchanged':True};rows.append(row);print(row,flush=True)
torch.npu.synchronize()
Path(__file__).with_name('floor_divide-fast-boundaries.json').write_text(json.dumps({'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'dependency_manifest':{str(Path(module.__file__).with_name(name)):hashlib.sha256(Path(module.__file__).with_name(name).read_bytes()).hexdigest() for name in ['_floor_divide.py','arithmetic_div.py','arithmetic.py']},'reference_sha256':hashlib.sha256(Path(__file__).with_name('floor_divide_reference.py').read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
