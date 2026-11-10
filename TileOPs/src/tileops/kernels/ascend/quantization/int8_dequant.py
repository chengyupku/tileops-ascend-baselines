"""Ascend INT8 dequantization: FP32 product followed by one output cast."""

# Registration is optional when the Ascend toolchain is unavailable.
try:
    import tilelang.ascend.language as T
except ImportError:
    pass
else:
    import functools
    from typing import ClassVar
    import torch
    import tilelang
    from tileops.backend import register_kernel_type
    from tileops.kernels.kernel_base import Kernel
    from tileops.kernels.quantization.call_spec import (
        INT8DequantPerTensorFwdInterface,
        INT8DequantPerChannelFwdInterface,
        INT8DequantPerBlockFwdInterface,
    )


    from ._dequant_tensor_striped import build as build_tensor_striped
    from ._dequant_rows_pipeline import build as build_rows_pipeline

    @functools.lru_cache(maxsize=256)
    def build(m,k,dtype,kind):
        width=1024
        chunks=(k+width-1)//width
        scale_cols=(k+127)//128
        scales=1 if kind=='tensor' else m if kind=='channel' else m*scale_cols
        blocks=min(64,m*chunks)
        @T.prim_func
        def main(Q:T.Tensor((m,k),'int8'),S:T.Tensor((scales,),'float32'),Y:T.Tensor((m,k),dtype)):
            with T.Kernel(blocks) as bx:
                q=T.alloc_shared((width,),'int8')
                s=T.alloc_shared((16,),'float32')
                y=T.alloc_shared((width,),dtype)
                for step in T.serial((m*chunks+blocks-1)//blocks):
                    job=step*blocks+bx
                    if job<m*chunks:
                        row=job//chunks
                        begin=(job%chunks)*width
                        valid=T.min(width,k-begin)
                        T.copy(Q[row,begin:begin+valid],q[:valid])
                        if kind=='tensor':
                            T.copy(S[:1],s[:1])
                        elif kind=='channel':
                            T.copy(S[row:row+1],s[:1])
                        else:
                            first=row*scale_cols+begin//128
                            count=(valid+127)//128
                            T.copy(S[first:first+count],s[:count])
                        with T.SimtVF(threads=1024):
                            for i in T.Parallel(width):
                                if i<valid:
                                    if kind=='block':
                                        y[i]=T.cast(q[i],'float32')*s[i//128]
                                    else:
                                        y[i]=T.cast(q[i],'float32')*s[0]
                        T.copy(y[:valid],Y[row,begin:begin+valid])
        return tilelang.compile(main,target='ascend',out_idx=[2])


    class _Dequant(Kernel):
        devices:ClassVar[frozenset[str]]=frozenset({'npu'})
        @classmethod
        def entry_for(cls,call):return call,lambda:cls(call)
        def __init__(self,call):
            super().__init__();self.call=call
            if call.out_dtype not in (torch.float16,torch.bfloat16,torch.float32):
                raise TypeError('Unsupported dequantization output dtype')
        def forward(self,q,scale):
            c=self.call
            if tuple(q.shape)!=(c.m,c.k) or q.dtype!=torch.int8 or scale.dtype!=torch.float32:
                raise ValueError('Invalid dequantization input signature')
            expected=1 if self.kind=='tensor' else c.m if self.kind=='channel' else c.m*((c.k+127)//128)
            if scale.numel()!=expected:raise ValueError('Invalid scale size')
            if not q.numel():return torch.empty_like(q,dtype=c.out_dtype)
            if self.kind=='tensor':
                self.compiled=build_tensor_striped(1,c.m*c.k,str(c.out_dtype).split('.')[-1],'tensor')
                return self.compiled(q.reshape(1,c.m*c.k),scale.reshape(-1)).reshape(c.m,c.k)
            builder=build_rows_pipeline if self.kind=='channel' else build
            self.compiled=builder(c.m,c.k,str(c.out_dtype).split('.')[-1],self.kind)
            return self.compiled(q,scale.reshape(-1))


    class AscendINT8DequantPerTensorKernel(_Dequant,INT8DequantPerTensorFwdInterface):
        kind='tensor'
    class AscendINT8DequantPerChannelKernel(_Dequant,INT8DequantPerChannelFwdInterface):
        kind='channel'
    class AscendINT8DequantPerBlockKernel(_Dequant,INT8DequantPerBlockFwdInterface):
        kind='block'

    register_kernel_type('INT8DequantPerTensorFwdOp','ascend',AscendINT8DequantPerTensorKernel)
    register_kernel_type('INT8DequantPerChannelFwdOp','ascend',AscendINT8DequantPerChannelKernel)
    register_kernel_type('INT8DequantPerBlockFwdOp','ascend',AscendINT8DequantPerBlockKernel)
