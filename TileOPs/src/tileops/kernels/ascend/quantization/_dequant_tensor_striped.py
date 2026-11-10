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


    @functools.lru_cache(maxsize=256)
    def build(m,k,dtype,kind):
        assert kind=='tensor'
        width=8192
        chunks=(k+width-1)//width
        scale_cols=(k+127)//128
        scales=1 if kind=='tensor' else m if kind=='channel' else m*scale_cols
        blocks=min(64,m*chunks)
        @T.prim_func
        def main(Q:T.Tensor((m,k),'int8'),S:T.Tensor((scales,),'float32'),Y:T.Tensor((m,k),dtype)):
            with T.Kernel(blocks) as bx:
                q=T.alloc_shared((width,),'int8')
                s=T.alloc_shared((64,),'float32')
                y=T.alloc_shared((width,),dtype)
                T.annotate_buffer_versions({q:2,y:2})
                T.copy(S[:1],s[:1])
                for step in T.Pipelined((m*chunks+blocks-1)//blocks,num_stages=2):
                    job=step*blocks+bx
                    if job<m*chunks:
                        row=job//chunks
                        begin=(job%chunks)*width
                        valid=T.min(width,k-begin)
                        T.copy(Q[row,begin:begin+valid],q[:valid])
                        if kind=='channel':
                            T.copy(S[row:row+1],s[:1])
                        elif kind=='block':
                            first=row*scale_cols+begin//128
                            count=(valid+127)//128
                            T.copy(S[first:first+count],s[:count])
                        with T.SimtVF(threads=1024):
                            for stripe in T.serial(width//1024):
                                for lane in T.Parallel(1024):
                                    i=stripe*1024+lane
                                    if i<valid:
                                        if kind=='block':
                                            y[i]=T.cast(q[i],'float32')*s[i//128]
                                        else:
                                            y[i]=T.cast(q[i],'float32')*s[0]
                        T.copy(y[:valid],Y[row,begin:begin+valid])
        return tilelang.compile(main,target='ascend',out_idx=[2])


