"""Use original BatchMatMulV3 as a rank-2 public matrix API via batch size1."""
import dataclasses
from pathlib import Path
import runpy
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_all_pairs.ops_nn import matrix,staging
from benchmarks.ascend_baselines import source_hashes,UnsupportedCase

def mm(binding,a,b,bias=None):
    import torch
    import torch.nn.functional as F
    m,k=a.shape;n=b.shape[1]
    lhs=F.pad(a.contiguous(),(0,-k%64,0,-m%1024))
    rhs=F.pad(b.T.contiguous(),(0,-k%64,0,-n%256))
    out=torch.empty((1,lhs.shape[0],rhs.shape[0]),device=a.device,dtype=a.dtype)
    binding.call('BatchMatMulV3',[lhs.unsqueeze(0),rhs.unsqueeze(0),None,None],[out],
                 {'adj_x1':False,'adj_x2':True,'enable_hf32':False})
    out=out.squeeze(0)[:m,:n].contiguous()
    if bias is not None:
        import ctypes as C
        y=torch.empty_like(out);bb=bias.expand_as(out).contiguous()
        alpha=torch.tensor(1.,device=a.device,dtype=torch.float32)
        binding.call('ForeachAddList',[out,bb,alpha],[y],{},names=['x10','x20','alpha','y0'],constants={2:(C.c_float*1)(1.)})
        out=y
    return out
matrix._mm=mm

def factory(case,library):
    if library!='ops-nn':raise ValueError('Only ops-nn')
    op=case._call.signature.name
    if op not in staging.CANDIDATES:raise UnsupportedCase('No existing original API composition')
    def build():
        c=staging.CANDIDATES[op](case)
        src=ROOT/'ascend_baselines/ops-nn/matmul/batch_mat_mul_v3'
        files=[p for p in src.rglob('*') if p.is_file() and p.suffix in ('.cpp','.h','.py','.json')]
        return dataclasses.replace(c,api=c.api.replace('MatMulV3','BatchMatMulV3(batch=1)'),
            adaptation=c.adaptation+' Rank2 multiply calls original BatchMatMulV3 with singleton batch; no modified MatMulV3 OpDef is executed.',
            source_sha256={**c.source_sha256,**source_hashes(Path(__file__),*files)})
    return build
registry.candidate_factory=factory
runpy.run_path(str(ROOT/'tools/measure_baseline_cache_pair.py'),run_name='__main__')
