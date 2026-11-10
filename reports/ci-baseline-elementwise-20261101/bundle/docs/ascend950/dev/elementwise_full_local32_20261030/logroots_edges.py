"""Public dispatch, independent CPU oracle, writes and input-sensitivity checks."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src')]
import torch
import torch_npu
import tilelang
from tileops.kernels.ascend.elementwise import _logroots as K
from tileops.ops import elementwise
torch.npu.Event=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('Event forbidden'))
p=argparse.ArgumentParser();p.add_argument('--op',required=True);p.add_argument('--output',required=True)
a=p.parse_args()
kind={'LogFwdOp':'log','Log1pFwdOp':'log1p','SqrtFwdOp':'sqrt','RsqrtFwdOp':'rsqrt'}[a.op]
importlib.import_module('tileops.kernels.ascend.elementwise.'+kind)
torch.npu.set_device(0)
torch.manual_seed(20261028)
rows=[]
def check(got,ref,dtype):
    tolerance={torch.float16:1e-3,torch.bfloat16:1.6e-2,torch.float32:1e-5}[dtype]
    torch.testing.assert_close(got,ref,rtol=tolerance,atol=tolerance,equal_nan=True)
    zero=ref==0
    assert torch.equal(torch.signbit(got[zero]),torch.signbit(ref[zero]))

for dtype in (torch.float16,torch.bfloat16,torch.float32):
    op=getattr(elementwise,a.op)()
    # Include a nearly full tail and an uneven grid pass with that tail: these
    # also exercise peak UB allocation after the persistent pipeline drains.
    shapes=[(0,5),(),(65,),(8193,),(16383,),(524289,),(540671,)]
    if kind=='log' and dtype==torch.bfloat16:
        shapes.extend([(129,),(193,)])
    for shape in shapes:
        x=(torch.rand(shape,dtype=torch.float32)*4+0.01).to(dtype)
        inp=x.to('npu');ref=getattr(torch,kind)(x.double()).to(dtype)
        got=op(inp).cpu();check(got,ref,dtype)
        assert torch.equal(inp.cpu().view(torch.uint8),x.view(torch.uint8)) if x.ndim else inp.cpu().item()==x.item()
        rows.append({'dtype':str(dtype),'shape':shape,'check':'public_entry','passed':True})
    # Unaligned tail, noncontiguous input and a second input prevent stale-output
    # and dispatch-only validation from masquerading as numerical correctness.
    x=(torch.rand((65,257))*2+0.01).to(dtype).T
    inp=x.to('npu');ref=getattr(torch,kind)(inp.cpu().double()).to(dtype)
    check(op(inp).cpu(),ref,dtype)
    changed=inp+1
    check(op(changed).cpu(),getattr(torch,kind)(changed.cpu().double()).to(dtype),dtype)
    rows.append({'dtype':str(dtype),'check':'noncontiguous_and_changed_input','passed':True})
    values=torch.tensor([float('nan'),float('inf'),-float('inf'),-2.,-1.,-.999,-.5,-1e-7,-1e-20,-1e-38,-1e-40,-0.,0.,1e-45,1e-40,1e-38,1.17549435e-38,1e-20,1e-7,.01,.5,1.,2.,1e4,1e20,3.4e38],dtype=dtype)
    x=values.repeat(400)[:8193].contiguous()
    inp=x.to('npu');got=op(inp).cpu();ref=getattr(torch,kind)(x.double()).to(dtype)
    check(got,ref,dtype)
    if kind=='log1p':
        tiny=x.abs()<5e-8
        assert torch.equal(got[tiny].view(torch.uint8),x[tiny].view(torch.uint8)), 'Tiny log1p inputs lost'
    npu_reference=getattr(torch,kind)(values.to('npu')).cpu()
    rows.append({'dtype':str(dtype),'check':'domain_and_subnormals','passed':True,
                 'samples':[{'input':str(v.item()),'cpu_fp64_oracle':str(r.item()),
                             'tileops':str(g.item()),'torch_npu_reference':str(t.item())}
                            for v,r,g,t in zip(values,ref,got,npu_reference)]})
    n=8193
    kernel=getattr(importlib.import_module('tileops.kernels.ascend.elementwise.'+kind),'Ascend'+a.op.removesuffix('FwdOp')+'Kernel')(n,dtype)
    explicit=tilelang.compile(kernel.compiled.prim_func,target='ascend',out_idx=[])
    # Use finite valid values so every NaN sentinel must be overwritten.
    x=(torch.rand(n)*3+.01).to(dtype);inp=x.to('npu')
    guarded=torch.full((n+128,),321.,device='npu',dtype=dtype)
    out=guarded[64:64+n];out.fill_(float('nan'))
    if hasattr(kernel,'lookup'):
        explicit(inp,kernel.lookup,out)
    else:
        explicit(inp,out)
    torch.npu.synchronize()
    assert bool((guarded[:64]==321.).all()) and bool((guarded[64+n:]==321.).all()), 'Out-of-bounds output write'
    actual=out.cpu();ref=getattr(torch,kind)(x.double()).to(dtype);check(actual,ref,dtype)
    for label,bad in [('unwritten_tail',actual.clone()),('wrong_shape',actual[:-1]),('wrong_value',actual.clone())]:
        if label=='unwritten_tail':bad[-1]=float('nan')
        if label=='wrong_value':bad[10]+=2
        try:check(bad,ref,dtype)
        except AssertionError:pass
        else:raise AssertionError('Checker accepted '+label)
    assert torch.equal(inp.cpu().view(torch.uint8),x.view(torch.uint8))
    rows.append({'dtype':str(dtype),'check':'sentinel_and_checker_negative_controls','passed':True})
    if kind=='log' and dtype==torch.bfloat16:
        all_values=torch.arange(65536,dtype=torch.int32,device='cpu').to(torch.int16).view(torch.bfloat16)
        all_values=torch.cat([all_values,all_values[:1]])
        all_input=all_values.to('npu')
        all_actual=op(all_input).cpu()
        all_reference=torch.log(all_values.double()).to(dtype)
        torch.testing.assert_close(all_actual,all_reference,rtol=0,atol=0,equal_nan=True)
        zero=all_reference==0
        assert torch.equal(torch.signbit(all_actual[zero]),torch.signbit(all_reference[zero]))
        assert torch.equal(all_input.cpu().view(torch.uint8),all_values.view(torch.uint8))
        table_hash=hashlib.sha256(kernel.lookup.cpu().view(torch.uint8).numpy().tobytes()).hexdigest() if hasattr(kernel,'lookup') else None
        rows.append({'dtype':str(dtype),'check':'all_65536_patterns_exact_CPU_and_tail','passed':True,
                     'table_sha256':table_hash})
    print('EDGES_DTYPE_PASSED',a.op,str(dtype),flush=True)
torch.npu.synchronize()
path=Path(a.output);path.parent.mkdir(exist_ok=True,parents=True)
module=importlib.import_module('tileops.kernels.ascend.elementwise.'+kind)
path.write_text(json.dumps({'op':a.op,'helper_sha256':hashlib.sha256(Path(K.__file__).read_bytes()).hexdigest(),
                           'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                           'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checks':rows},indent=2))
print('EDGES_PASSED',a.op,len(rows),flush=True)
