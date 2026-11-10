"""Real TileLang autotuning with isolated msprof windows; no Event timing."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src')]
import torch
import torch_npu
def forbidden(*a,**kw):raise AssertionError('Event timing forbidden')
torch.npu.Event=forbidden
from tileops.kernels.ascend.elementwise import _logroots as K
from benchmarks.api import cases
from workloads.device import set_run_device
from autotune_compat import run_search
N=0;DTYPE='float16';KIND='log'

def compile_fn(tile,blocks,stages):
    return K.build(N,DTYPE,KIND,tile,blocks,stages)

def factory(tile,blocks,stages):return compile_fn(tile,blocks,stages).prim_func

def main():
    global N,DTYPE,KIND
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--output',required=True);p.add_argument('--op',required=True)
    a=p.parse_args();torch.npu.set_device(0);set_run_device('npu');torch.manual_seed(20261028)
    KIND={'LogFwdOp':'log','Log1pFwdOp':'log1p','SqrtFwdOp':'sqrt','RsqrtFwdOp':'rsqrt'}[a.op]
    case=next(c for c in cases(a.op) if c.id==a.case)
    xs=case.inputs;N=xs[0].numel();DTYPE=str(xs[0].dtype).removeprefix('torch.')
    reference=case.reference(*(x.cpu() for x in xs)).to('npu').reshape(-1)
    inputs=[x.reshape(-1) for x in xs]
    control=dict(tile=8192,blocks=64,stages=2)
    specs=[(8192,64,2),(4096,64,3),(16384,64,2),(8192,72,2),(8192,64,3)]
    configs=[dict(tile=t,blocks=b,stages=s) for t,b,s in specs if 2*t*s*xs[0].element_size()<=248*1024]
    result=run_search(factory=factory,configs=configs,compile_fn=compile_fn,
        supply_prog=lambda params:inputs,reference_prog=lambda *args:reference,
        output_dir=a.output,source_paths=[Path(__file__),Path(K.__file__)],control_config=control,
        warmup=3,repeats=15,search_backend='profile_short',profile_warmup=1,profile_repeats=3,
        rtol={'float32':1e-5,'float16':1e-3,'bfloat16':1.6e-2}[DTYPE],atol={'float32':1e-5,'float16':1e-3,'bfloat16':1.6e-2}[DTYPE])
    print(json.dumps({k:v for k,v in result.items() if k!='trials'},default=str),flush=True)
if __name__=='__main__':main()
