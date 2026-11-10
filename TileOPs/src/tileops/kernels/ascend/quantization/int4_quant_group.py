"""Bounded two-pass asymmetric INT4 group quantization for Ascend."""

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
    from tileops.kernels.quantization.call_spec import INT4QuantPerGroupFwdInterface

    @functools.lru_cache(maxsize=256)
    def build(total,group_size,dtype):
        groups=total//group_size
        width=min(1024,tilelang.next_power_of_2(group_size))
        blocks=min(64,groups)
        @T.prim_func
        def main(X:T.Tensor((total,),dtype),Q:T.Tensor((total//2,),'int8'),P:T.Tensor((groups,2),'float32')):
            with T.Kernel(blocks) as bx:
                x=T.alloc_shared((width,),dtype)
                q=T.alloc_shared((width//2,),'int8')
                stats=T.alloc_shared((16,),'float32')
                params=T.alloc_shared((16,),'float32')
                for step in T.serial((groups+blocks-1)//blocks):
                    group=step*blocks+bx
                    if group<groups:
                        with T.SimtVF(threads=32):
                            stats[0]=T.float32(float('inf'))
                            stats[1]=T.float32(-float('inf'))
                        for chunk in T.serial((group_size+width-1)//width):
                            offset=chunk*width
                            valid=T.min(width,group_size-offset)
                            T.copy(X[group*group_size+offset:group*group_size+offset+valid],x[:valid])
                            with T.SimtVF(threads=min(256,width)):
                                lo=T.alloc_reducer((1,),'float32',op='min')
                                hi=T.alloc_reducer((1,),'float32',op='max')
                                T.reducer_init(lo);T.reducer_init(hi)
                                for i in T.Parallel(width):
                                    if i<valid:
                                        T.reducer_update(lo[0],T.cast(x[i],'float32'))
                                        T.reducer_update(hi[0],T.cast(x[i],'float32'))
                                low=T.alloc_fragment((1,),'float32')
                                high=T.alloc_fragment((1,),'float32')
                                T.finalize_reducer(lo,low);T.finalize_reducer(hi,high)
                                stats[0]=T.min(stats[0],low[0])
                                stats[1]=T.max(stats[1],high[0])
                        with T.SimtVF(threads=32):
                            span=stats[1]-stats[0]
                            estimate=T.float32(16)/span
                            residual=T.call_extern('float32','fmaf',-estimate,span,T.float32(16))
                            multiplier=estimate+residual/span
                            stats[2]=T.if_then_else(stats[0]==stats[1],T.float32(1),multiplier)
                            stats[3]=(stats[0]+stats[1])*T.float32(0.5)
                            reciprocal=T.float32(1)/stats[2]
                            reciprocal_residual=T.call_extern('float32','fmaf',-reciprocal,stats[2],T.float32(1))
                            params[0]=reciprocal+reciprocal_residual/stats[2]
                            params[1]=stats[3]
                        T.copy(params[:2],P[group,:2])
                        for chunk in T.serial((group_size+width-1)//width):
                            offset=chunk*width
                            valid=T.min(width,group_size-offset)
                            T.copy(X[group*group_size+offset:group*group_size+offset+valid],x[:valid])
                            with T.SimtVF(threads=min(256,width//2)):
                                for i in T.Parallel(width//2):
                                    if 2*i<valid:
                                        first=T.nearbyint((T.cast(x[2*i],'float32')-stats[3])*stats[2])
                                        second=T.nearbyint((T.cast(x[2*i+1],'float32')-stats[3])*stats[2])
                                        a=T.cast(T.min(T.max(first,T.float32(-8)),T.float32(7)),'int32')
                                        b=T.cast(T.min(T.max(second,T.float32(-8)),T.float32(7)),'int32')
                                        q[i]=T.cast(T.bitwise_or(T.shift_left(a,4),T.bitwise_and(b,15)),'int8')
                            T.copy(q[:valid//2],Q[(group*group_size+offset)//2:(group*group_size+offset+valid)//2])
        return tilelang.compile(main,target='ascend',out_idx=[])

    class AscendINT4QuantPerGroupKernel(Kernel,INT4QuantPerGroupFwdInterface):
        devices:ClassVar[frozenset[str]]=frozenset({'npu'})
        @classmethod
        def entry_for(cls,call):return call,lambda:cls(call)
        def __init__(self,call):
            super().__init__();self.call=call
        def forward(self,x):
            c=self.call;g=c.group_size
            if not g or g%2 or c.cols%g:raise ValueError('Even group size must divide columns')
            if tuple(x.shape)!=(c.rows,c.cols) or x.dtype!=c.dtype:raise ValueError('Invalid input signature')
            if not x.numel():return torch.empty((c.rows,c.cols//2),device=x.device,dtype=torch.int8),torch.empty((0,2),device=x.device,dtype=torch.float32)
            self.compiled=build(x.numel(),g,str(c.dtype).split('.')[-1])
            q=torch.empty((x.numel()//2,),dtype=torch.int8,device=x.device)
            params=torch.empty((x.numel()//g,2),dtype=torch.float32,device=x.device)
            self.compiled(x.reshape(-1),q,params)
            return q.reshape(c.rows,c.cols//2),params
    register_kernel_type('INT4QuantPerGroupFwdOp','ascend',AscendINT4QuantPerGroupKernel)
