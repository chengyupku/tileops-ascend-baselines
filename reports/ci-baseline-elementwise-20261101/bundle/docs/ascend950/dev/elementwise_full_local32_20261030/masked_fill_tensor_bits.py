"""Bit-exact masked fill, including NaN payloads, int64, and noncanonical bool masks."""
import json,hashlib
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import masked_fill_tensor as module
from tileops.ops.elementwise import MaskedFillTensorFwdOp
DTYPES=[torch.bool,torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64,torch.float16,torch.bfloat16,torch.float32]
rows=[];torch.manual_seed(42);n=65537
for dtype in DTYPES:
 size=dtype.itemsize;storage={1:torch.int8,2:torch.int16,4:torch.int32,8:torch.int64}[size]
 if size<=2:raw=torch.arange(n,dtype=torch.int64).to(storage)
 elif size==4:raw=torch.randint(-2147483648,2147483647,(n,),dtype=torch.int32)
 else:
  raw=torch.randint(-2147483648,2147483647,(n,),dtype=torch.int64)*4294967296+torch.randint(0,4294967295,(n,),dtype=torch.int64)
 x=raw.view(dtype);a=raw.npu().view(dtype);op=MaskedFillTensorFwdOp()
 fills=[torch.iinfo(storage).min,torch.iinfo(storage).max,1]
 for mode in ['all_false','all_true','mixed_raw']:
  maskraw=(torch.zeros(n,dtype=torch.uint8) if mode=='all_false' else torch.full((n,),255,dtype=torch.uint8) if mode=='all_true' else torch.tensor([0,1,2,127,128,255],dtype=torch.uint8).repeat((n+5)//6)[:n])
  mask=maskraw.npu().view(torch.bool)
  for fv in fills:
   scalar=torch.tensor(fv,dtype=storage);value=scalar.npu().view(dtype)
   actual=op(a,mask,value).view(storage).cpu()
   expected=torch.where(maskraw!=0,scalar,raw)
   assert torch.equal(actual,expected),f'{dtype} {mode} {fv}: bit mismatch'
   assert torch.equal(a.view(storage).cpu(),raw) and torch.equal(mask.view(torch.uint8).cpu(),maskraw)
   rows.append({'dtype':str(dtype),'mode':mode,'fill_bits':fv,'values':n,'passed':True});print(rows[-1],flush=True)
 # Bidirectional broadcast: one input row and two mask rows; second uses a raw true byte.
 maskraw=torch.tensor([[0],[128]],dtype=torch.uint8);mask=maskraw.npu().view(torch.bool)
 scalar=torch.tensor(fills[1],dtype=storage);value=scalar.npu().view(dtype)
 actual=op(a.reshape(1,n),mask,value).view(storage).cpu();expected=torch.where(maskraw!=0,scalar,raw.reshape(1,n))
 assert torch.equal(actual,expected)
 rows.append({'dtype':str(dtype),'mode':'broadcast_raw','fill_bits':fills[1],'values':2*n,'passed':True});print(rows[-1],flush=True)
torch.npu.synchronize()
Path(__file__).with_name('masked-fill-tensor-bits.json').write_text(json.dumps({'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
