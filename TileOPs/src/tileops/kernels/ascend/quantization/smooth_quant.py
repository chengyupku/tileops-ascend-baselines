"""Ascend SmoothQuant with FP32 smoothing and per-row INT8 scales."""

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
    from tileops.kernels.quantization.call_spec import SmoothQuantFwdInterface

    @functools.lru_cache(maxsize=256)
    def build(m,k,dtype):
        group_width=k
        groups_per_row=(k+group_width-1)//group_width
        groups=m*groups_per_row
        width=min(1024,tilelang.next_power_of_2(group_width))
        pitch=max(16,width)
        blocks=min(64,groups)
        @T.prim_func
        def main(X:T.Tensor((m,k),dtype),Smooth:T.Tensor((k,),'float32'),Q:T.Tensor((m,k),'int8'),S:T.Tensor((m,groups_per_row),'float32')):
            with T.Kernel(blocks) as bx:
                x=T.alloc_shared((pitch,),dtype)
                q=T.alloc_shared((pitch,),'int8')
                smooth=T.alloc_shared((pitch,),'float32')
                stats=T.alloc_shared((16,),'float32')
                for step in T.serial((groups+blocks-1)//blocks):
                    job=step*blocks+bx
                    if job<groups:
                        row=job//groups_per_row
                        group=job%groups_per_row
                        start=group*group_width
                        valid_group=T.min(group_width,k-start)
                        T.clear(stats)
                        for chunk in T.serial((group_width+width-1)//width):
                            offset=chunk*width
                            valid=T.min(width,valid_group-offset)
                            T.copy(X[row,start+offset:start+offset+valid],x[:valid])
                            T.copy(Smooth[start+offset:start+offset+valid],smooth[:valid])
                            with T.SimtVF(threads=min(256,width)):
                                peak=T.alloc_reducer((1,),'float32',op='max')
                                T.reducer_init(peak)
                                nan_seen=T.alloc_reducer((1,),'float32',op='max')
                                T.reducer_init(nan_seen)
                                for i in T.Parallel(width):
                                    if i<valid:
                                        numerator=T.cast(x[i],'float32')
                                        estimate=numerator/smooth[i]
                                        residual=T.call_extern('float32','fmaf',-estimate,smooth[i],numerator)
                                        value=estimate+residual/smooth[i]
                                        T.reducer_update(peak[0],T.if_then_else(T.isnan(value),T.float32(0),T.abs(value)))
                                        T.reducer_update(nan_seen[0],T.if_then_else(T.isnan(value),T.float32(1),T.float32(0)))
                                result=T.alloc_fragment((1,),'float32')
                                T.finalize_reducer(peak,result)
                                stats[0]=T.max(stats[0],result[0])
                                nan_result=T.alloc_fragment((1,),'float32')
                                T.finalize_reducer(nan_seen,nan_result)
                                stats[1]=T.max(stats[1],nan_result[0])
                        with T.SimtVF(threads=32):
                            estimate=stats[0]/T.float32(127)
                            residual=T.call_extern('float32','fmaf',-estimate,T.float32(127),stats[0])
                            corrected=estimate+residual/T.float32(127)
                            stats[0]=T.if_then_else(stats[1]>0,T.float32(1),T.if_then_else(stats[0]==0,T.float32(1),corrected))
                        T.copy(stats[:1],S[row,group:group+1])
                        for chunk in T.serial((group_width+width-1)//width):
                            offset=chunk*width
                            valid=T.min(width,valid_group-offset)
                            T.copy(X[row,start+offset:start+offset+valid],x[:valid])
                            T.copy(Smooth[start+offset:start+offset+valid],smooth[:valid])
                            with T.SimtVF(threads=width):
                                for i in T.Parallel(width):
                                    if i<valid:
                                        numerator=T.cast(x[i],'float32')
                                        estimate=numerator/smooth[i]
                                        residual=T.call_extern('float32','fmaf',-estimate,smooth[i],numerator)
                                        value=estimate+residual/smooth[i]
                                        quotient=value/stats[0]
                                        remainder=T.call_extern('float32','fmaf',-quotient,stats[0],value)
                                        rounded=T.nearbyint(quotient+remainder/stats[0])
                                        q[i]=T.cast(T.min(T.max(rounded,T.float32(-127)),T.float32(127)),'int8')
                            T.copy(q[:valid],Q[row,start+offset:start+offset+valid])
        return tilelang.compile(main,target='ascend',out_idx=[2,3])

    class AscendSmoothQuantKernel(Kernel,SmoothQuantFwdInterface):
        devices:ClassVar[frozenset[str]]=frozenset({'npu'})
        @classmethod
        def entry_for(cls,call):return call,lambda:cls(call)
        def __init__(self,call):
            super().__init__();self.call=call
            if call.dtype not in (torch.float16,torch.bfloat16,torch.float32):raise TypeError('Unsupported quantization dtype')
        def forward(self,x,smooth):
            c=self.call
            if tuple(x.shape)!=(c.rows,c.cols) or x.dtype!=c.dtype:raise ValueError('Invalid input signature')
            if tuple(smooth.shape)!=(c.cols,) or smooth.dtype!=torch.float32 or smooth.device!=x.device:raise ValueError('Invalid smoothing factors')
            if c.rows==0:
                return torch.empty_like(x,dtype=torch.int8),torch.empty((0,),device=x.device,dtype=torch.float32)
            if c.cols<=0:raise ValueError('Quantization requires nonempty rows')
            self.compiled=build(c.rows,c.cols,str(c.dtype).split('.')[-1])
            q,scale=self.compiled(x,smooth)
            return q,scale.reshape(c.rows)
    register_kernel_type('SmoothQuantFwdOp','ascend',AscendSmoothQuantKernel)
