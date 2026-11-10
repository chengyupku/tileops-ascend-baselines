"""Compositions of unmodified, owned TileLang-Ascend builders; lazy device loading."""
from math import prod
from pathlib import Path
import hashlib

from . import native
from .plans import ELEMENTWISE
from .omission_layout import build_run
from .omission_contracts import eligible


class Engine:
    def __init__(self):
        self.kernels = {}

    def _kernel(self,name,*args,**kwargs):
        key=(name,args,tuple(sorted(kwargs.items())))
        if key not in self.kernels:
            src = 'examples/normalization/rms_norm.py' if name=='_rms_norm' else (
                'examples/gemm/example_gemm_pto_developer.py' if name=='matmul' else ELEMENTWISE)
            module=native.load(src,name,'pass_configs')
            self.kernels[key]=native.compile_builder(module,name,*args,**kwargs,out_idx=[-1])
        return self.kernels[key]

    def _vector(self,name,*values,dt='float32',compare=False):
        import torch
        shape=tuple(torch.broadcast_shapes(*(v.shape for v in values))); count=prod(shape)
        rows=((count+4095)//4096)*64; cols=64
        dtype=getattr(torch,dt)
        args=[rows,cols,64,64]
        options={'dtype':dt}
        if compare:
            args.append('NE'); options['out_dtype']='uint8'
        kernel=self._kernel(name,*args,**options)
        inputs=[]
        for value in values:
            packed=torch.zeros((rows*cols,),dtype=dtype,device=value.device)
            packed[:count].copy_(value.expand(shape).contiguous().reshape(-1).to(dtype))
            inputs.append(packed.reshape(rows,cols))
        result=kernel(*inputs)
        if compare:
            # Decode the native bit mask; these shifts change representation only.
            packed_bits=result.to(torch.int32).reshape(-1)
            result=torch.stack([(packed_bits>>bit)&1 for bit in range(8)],-1).reshape(-1)
        return result.reshape(-1)[:count].reshape(shape)

    def binary(self,op,a,b):
        name={'add':'vec_add_developer','sub':'vec_sub','mul':'vec_mul',
              'div':'vec_div','min':'vec_min','max':'vec_max'}[op]
        return self._vector(name,a,b)

    def unary(self,op,x):
        return self._vector({'sqrt':'sqrt','abs':'vec_abs'}[op],x)

    def truth(self,x):
        import torch
        return self._vector('compare',x,torch.zeros((),dtype=torch.float32,device=x.device),compare=True)

    def logical(self,op,a,b):
        return self._vector('bitwise_'+op,a,b,dt='int16')

    def reduce(self,matrix,op):
        import torch
        rows,width=matrix.shape
        if not 0<width<=32768: raise ValueError('Whole-row reduction width outside audited bound')
        strip=max(1,min(16,16384//width))
        kernel=self._kernel('reduce_runtime_semantics_kernel',strip,width,op,-1,dtype='float32')
        output=torch.empty(rows,dtype=torch.float32,device=matrix.device)
        for start in range(0,rows,strip):
            count=min(strip,rows-start)
            packed=torch.zeros((strip,width),dtype=torch.float32,device=matrix.device)
            packed[:count].copy_(matrix[start:start+count])
            output[start:start+count].copy_(kernel(packed)[:count])
        return output

    def rms(self,matrix,eps):
        import torch
        rows,width=matrix.shape; padded=(rows+63)//64*64
        kernel=self._kernel('_rms_norm',padded,width,64,64,eps,dtype='float32')
        packed=torch.zeros((padded,width),dtype=torch.float32,device=matrix.device)
        packed[:rows].copy_(matrix)
        return kernel(packed)[:rows]

    def gemm(self,a,b):
        import torch.nn.functional as F
        m,k=a.shape; n=b.shape[1]
        mp=(m+127)//128*128; np=(n+255)//256*256; kp=(k+63)//64*64
        kernel=self._kernel('matmul',mp,np,kp,128,256,64,dtype=str(a.dtype).split('.')[-1])
        return kernel(F.pad(a,(0,kp-k,0,mp-m)),F.pad(b,(0,np-n,0,kp-k)))[:m,:n]


def make(case, *, op):
    from benchmarks.ascend_baselines import UnsupportedCase
    if not eligible(op,case.params): raise UnsupportedCase('Outside the reviewed omission contracts')
    engine=Engine()
    run=build_run(op,case,engine)
    sources=[ELEMENTWISE,'tilelang/language/ascend_tile.py']
    if 'RMSNorm' in op: sources.append('examples/normalization/rms_norm.py')
    if op=='SSDChunkCouplingFwdOp': sources.append('examples/gemm/example_gemm_pto_developer.py')
    candidate=native.candidate(run,'owned-original API composition / '+op,
        'FP32 native intermediate arithmetic; complete original reduction domains and GEMM K; '
        'bounded independent rows/windows; all packing and metadata reads inside run. '
        'FusedAdd casts its complete residual before norm; all declared outputs retained.',*sources)
    for name in ('omissions.py','omission_layout.py','omission_contracts.py','omission_cases.json'):
        path=Path(__file__).with_name(name)
        candidate.source_sha256[str(path.relative_to(native.WORKSPACE))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return candidate
