"""Detailed pipe analysis, calling low-level msprof directly (no event estimate)."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'TileOPs/src'),str(ROOT/'TileOPs')]
import torch
import torch_npu

from tools.measure_baselines import _get_flush_kernel_and_buffer
from tilelang.profiler.msprof import bench_with_msprof

def forbidden(*args,**kwargs):raise AssertionError('Event is forbidden, including runtime estimates')
torch.npu.Event=forbidden
parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--op',required=True);parser.add_argument('--regex',default='16M-float32');args=parser.parse_args()
K=__import__('importlib').import_module('tileops.kernels.ascend.elementwise.'+{'LogFwdOp':'log','Log1pFwdOp':'log1p','SqrtFwdOp':'sqrt','RsqrtFwdOp':'rsqrt'}[args.op])
from benchmarks.api import cases
from workloads.device import set_run_device
set_run_device('npu');torch.npu.set_device(0)
output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
source_hash=hashlib.sha256(Path(K.__file__).read_bytes()).hexdigest()
flush,cache=_get_flush_kernel_and_buffer(torch,torch_npu)
for case in cases(args.op):
    if not __import__('re').search(args.regex,case.id):continue
    label=case.id;a=case.inputs[0]
    op=getattr(K,'Ascend'+args.op.removesuffix('FwdOp')+'Kernel')(a.numel(),a.dtype)
    tol={torch.float16:1e-3,torch.bfloat16:1.6e-2,torch.float32:1e-5}[a.dtype]
    actual=op(a).cpu();reference=case.reference(a.cpu())
    torch.testing.assert_close(actual,reference,rtol=tol,atol=tol,equal_nan=True)
    (output/(label+'.c')).write_text(op.compiled.get_kernel_source())
    for _ in range(3):flush(cache);torch.npu.synchronize();op(a);torch.npu.synchronize()
    result=bench_with_msprof(lambda:op(a),cache,15,detailed=True)
    row={'purpose':'diagnostic_only_not_formal_M4','backend':'msprof_PipeUtilization','event_used':False,'warmup':3,'repeats':15,'flush_mib':1024,'correctness':'CPU oracle dtype tolerance','correctness_tolerance':{'rtol':tol,'atol':tol},'max_abs_error':float((actual-reference).abs().max()),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':source_hash,'helper_sha256':hashlib.sha256(Path(K.__file__).with_name('_logroots.py').read_bytes()).hexdigest(),'label':label,'logical_bytes':(a.numel()*2)*a.element_size(),'profile':dataclasses.asdict(result),'note':'Pipe utilization is busy-cycle fraction; not bandwidth or peak FLOPS utilization.'}
    (output/(label+'.json')).write_text(json.dumps(row,indent=2)+'\n');print(label,row,flush=True)
assert hashlib.sha256(Path(K.__file__).read_bytes()).hexdigest()==source_hash
