"""Exhaustive low-dtype bit patterns and sampled FP32 domain validation."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src')]
import torch
import torch_npu
from candidate import _logroots_v12 as K
torch.npu.Event=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('Event forbidden'))
torch.npu.set_device(0)
torch.manual_seed(20261028)
rows=[]
for dtype in (torch.bfloat16,):
    name=str(dtype).removeprefix('torch.')
    if dtype!=torch.float32:
        x=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(dtype)
        x=torch.cat([x,x[:1]])
    else:
        bits=torch.randint(-2147483648,2147483647,(131072,),dtype=torch.int32)
        small=torch.linspace(-.02,.02,131072)
        x=torch.cat([bits.view(torch.float32),small,torch.tensor([-0.])])
    inp=x.to('npu');fn=K.build(x.numel(),name,'log')
    got=fn(inp).cpu();ref=torch.log(x.double()).to(dtype)
    tol={torch.float16:1e-3,torch.bfloat16:1.6e-2,torch.float32:1e-5}[dtype]
    torch.testing.assert_close(got,ref,rtol=tol,atol=tol,equal_nan=True)
    zeros=ref==0
    assert torch.equal(torch.signbit(got[zeros]),torch.signbit(ref[zeros]))
    assert torch.equal(inp.cpu().view(torch.uint8),x.view(torch.uint8))
    finite=torch.isfinite(ref)&(ref!=0)
    error=(got[finite].double()-ref[finite].double()).abs()
    row={'dtype':name,'elements':x.numel(),'passed':True,'all_bit_patterns':dtype!=torch.float32,
         'max_abs_error':float(error.max()),'max_relative_error':float((error/ref[finite].double().abs()).max()),
         'source_sha256':hashlib.sha256(Path(K.__file__).read_bytes()).hexdigest()}
    rows.append(row);print(json.dumps(row),flush=True)
torch.npu.synchronize()
out=Path(__file__).resolve().parent/'edges/log-bf16-v12-exhaustive.json'
out.write_text(json.dumps(rows,indent=2))
