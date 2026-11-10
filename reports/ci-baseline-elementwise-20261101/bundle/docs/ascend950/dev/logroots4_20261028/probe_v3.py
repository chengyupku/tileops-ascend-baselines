"""Numerical and output-coverage probe, including non-tile-aligned shapes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT/'TileOPs'), str(ROOT/'TileOPs/src')]
import torch
import torch_npu
from candidate import _logroots_v2 as K
torch.npu.Event = lambda *a, **kw: (_ for _ in ()).throw(AssertionError('Event forbidden'))
torch.npu.set_device(0)
p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
p.add_argument('--dtype',default='float32')
p.add_argument('--n',type=int,default=4097)
a=p.parse_args()
dt=getattr(torch,a.dtype)
values=[float('nan'),float('inf'),-float('inf'),-2.,-1.,-0.99999994,-0.5,-0.01,-1e-4,-1e-7,-1e-20,-1e-38,-1e-40,-0.,0.,1e-45,1e-40,1e-38,1.17549435e-38,1e-20,1e-10,1e-7,1e-4,0.01,0.1,0.5,1.,2.,3.,1e4,1e10,1e20,3.4e38]
base=torch.tensor(values,dtype=dt)
x=base.repeat((a.n+len(values)-1)//len(values))[:a.n].contiguous()
rows=[]
for kind in ('log','log1p','sqrt','rsqrt'):
    row={'kind':kind,'dtype':a.dtype,'n':a.n,'source':hashlib.sha256(Path(K.__file__).read_bytes()).hexdigest()}
    try:
        fn=K.build(a.n,a.dtype,kind)
        xn=x.to('npu');before=xn.cpu().view(torch.uint8).clone()
        out=torch.full_like(xn,12345.)
        import tilelang
        explicit=tilelang.compile(fn.prim_func,target="ascend",out_idx=[])
        explicit(xn,out)
        torch.npu.synchronize()
        got=out.cpu();ref=getattr(torch,kind)(x.double()).to(dt)
        tol={torch.float32:1e-5,torch.float16:1e-3,torch.bfloat16:1.6e-2}[dt]
        close=torch.isclose(got,ref,atol=tol,rtol=tol,equal_nan=True)
        zero=ref==0
        sign_ok=torch.equal(torch.signbit(got[zero]),torch.signbit(ref[zero]))
        row.update(passed=bool(close.all()) and sign_ok,zero_sign_ok=sign_ok,
                   mismatches=int((~close).sum()),samples=[{'x':float(x[i]),'got':float(got[i]),'ref':float(ref[i])} for i in range(min(len(values),a.n))],
                   input_unchanged=torch.equal(before,xn.cpu().view(torch.uint8)))
        (Path(a.output).parent/(kind+'-'+a.dtype+'-'+str(a.n)+'.c')).write_text(fn.get_kernel_source())
    except Exception as e:
        import traceback
        row.update(passed=False,error=traceback.format_exc())
    rows.append(row)
    print(json.dumps(row),flush=True)
Path(a.output).write_text(json.dumps(rows,indent=2))
torch.npu.synchronize()
raise SystemExit(0 if all(r['passed'] for r in rows) else 1)
