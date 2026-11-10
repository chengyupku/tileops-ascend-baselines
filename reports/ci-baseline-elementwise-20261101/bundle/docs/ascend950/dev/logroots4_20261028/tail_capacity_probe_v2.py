"""Early worst-tail/UB check for the shared FP32 three-stage pipeline."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src')]
import torch
import torch_npu
import tilelang
from candidate import _logroots_v7 as K
torch.npu.Event=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('Event forbidden'))
torch.npu.set_device(0)
rows=[]
for n in (16383,540671):
    x=torch.linspace(.01,3,n,dtype=torch.float32)
    inp=x.to('npu');fn=K.build(n,'float32','log1p',8192,64,3)
    explicit=tilelang.compile(fn.prim_func,target='ascend',out_idx=[])
    guarded=torch.full((n+128,),321.,dtype=torch.float32,device='npu')
    out=guarded[64:64+n];out.fill_(float('nan'));explicit(inp,out);torch.npu.synchronize()
    assert bool((guarded[:64]==321.).all()) and bool((guarded[64+n:]==321.).all())
    torch.testing.assert_close(out.cpu(),torch.log1p(x.double()).float(),rtol=1e-5,atol=1e-5)
    assert torch.equal(inp.cpu(),x)
    row={'n':n,'config':[8192,64,3],'passed':True,'helper_sha256':hashlib.sha256(Path(K.__file__).read_bytes()).hexdigest()}
    rows.append(row);print(json.dumps(row),flush=True)
torch.npu.synchronize()
(Path(__file__).resolve().parent/'edges/tail-capacity-v2.json').write_text(json.dumps(rows,indent=2))
