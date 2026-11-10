"""Ascend FP8 row quantization with the contract amax floor."""

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
    from tileops.kernels.quantization.fp8_quant import FP8QuantFwdInterface

    @functools.lru_cache(maxsize=256)
    def build_batched(m,k,dtype):
        assert 0<k<=1024
        width=tilelang.next_power_of_2(k)
        rows=max(1,min(16,1024//width))
        pitch=max(16,width)
        tiles=(m+rows-1)//rows
        blocks=min(64,tiles)
        @T.prim_func
        def main(X:T.Tensor((m,k),dtype),Q:T.Tensor((m,k),'float8_e4m3fn'),S:T.Tensor((m,1),'float32')):
            with T.Kernel(blocks) as bx:
                x=T.alloc_shared((rows,pitch),dtype)
                q=T.alloc_shared((rows,pitch),'float8_e4m3fn')
                s=T.alloc_shared((rows,1),'float32')
                for step in T.serial((tiles+blocks-1)//blocks):
                    job=step*blocks+bx
                    if job<tiles:
                        start=job*rows
                        valid=T.min(rows,m-start)
                        T.copy(X[start:start+valid,:k],x[:valid,:k])
                        with T.SimtVF(threads=256):
                            peak=T.alloc_reducer((rows,),'float32',op='max')
                            T.reducer_init(peak)
                            for r,i in T.Parallel(rows,width):
                                if r<valid and i<k:
                                    T.reducer_update(peak[r],T.abs(T.cast(x[r,i],'float32')))
                            maximum=T.alloc_fragment((rows,),'float32')
                            T.finalize_reducer(peak,maximum)
                            scales=T.alloc_fragment((rows,),'float32')
                            for r in T.Parallel(rows):
                                scales[r]=T.max(maximum[r],T.float32(1e-4))/T.float32(448)
                                s[r,0]=scales[r]
                            for r,i in T.Parallel(rows,width):
                                if r<valid and i<k:
                                    scaled=T.cast(x[r,i],'float32')/scales[r]
                                    q[r,i]=T.cast(T.min(T.max(scaled,T.float32(-448)),T.float32(448)),'float8_e4m3fn')
                        T.copy(q[:valid,:k],Q[start:start+valid,:k])
                        T.copy(s[:valid,:],S[start:start+valid,:])
        return tilelang.compile(main,target='ascend',out_idx=[1,2])

    @functools.lru_cache(maxsize=256)
    def build(m,k,dtype):
        if 0 < k <= 1024:
            return build_batched(m,k,dtype)
        group_width=k
        groups_per_row=(k+group_width-1)//group_width
        groups=m*groups_per_row
        width=min(1024,tilelang.next_power_of_2(group_width))
        pitch=max(16,width)
        blocks=min(64,groups)
        @T.prim_func
        def main(X:T.Tensor((m,k),dtype),Q:T.Tensor((m,k),'float8_e4m3fn'),S:T.Tensor((m,groups_per_row),'float32')):
            with T.Kernel(blocks) as bx:
                x=T.alloc_shared((pitch,),dtype)
                q=T.alloc_shared((pitch,),'float8_e4m3fn')
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
                            with T.SimtVF(threads=min(256,width)):
                                peak=T.alloc_reducer((1,),'float32',op='max')
                                T.reducer_init(peak)
                                for i in T.Parallel(width):
                                    if i<valid:T.reducer_update(peak[0],T.abs(T.cast(x[i],'float32')))
                                result=T.alloc_fragment((1,),'float32')
                                T.finalize_reducer(peak,result)
                                stats[0]=T.max(stats[0],result[0])
                        with T.SimtVF(threads=32):
                            stats[0]=T.max(stats[0],T.float32(1e-4))/T.float32(448)
                        T.copy(stats[:1],S[row,group:group+1])
                        for chunk in T.serial((group_width+width-1)//width):
                            offset=chunk*width
                            valid=T.min(width,valid_group-offset)
                            T.copy(X[row,start+offset:start+offset+valid],x[:valid])
                            with T.SimtVF(threads=min(256,width)):
                                for i in T.Parallel(width):
                                    if i<valid:
                                        scaled=T.cast(x[i],'float32')/stats[0]
                                        q[i]=T.cast(T.min(T.max(scaled,T.float32(-448)),T.float32(448)),'float8_e4m3fn')
                            T.copy(q[:valid],Q[row,start+offset:start+offset+valid])
        return tilelang.compile(main,target='ascend',out_idx=[1,2])

    class AscendFP8QuantKernel(Kernel,FP8QuantFwdInterface):
        devices:ClassVar[frozenset[str]]=frozenset({'npu'})
        @classmethod
        def entry_for(cls,call):return call,lambda:cls(call)
        def __init__(self,call):
            super().__init__();self.call=call
        def forward(self,x):
            c=self.call
            shape=(c.batch,c.seq_len_kv,c.kv_group,c.index_dim)
            if tuple(x.shape)!=shape or x.dtype!=c.dtype:raise ValueError('Invalid FP8 quantization input')
            rows=c.batch*c.seq_len_kv*c.kv_group
            if not rows:
                return torch.empty(shape[:-1],device=x.device,dtype=torch.float32),torch.empty_like(x,dtype=torch.float8_e4m3fn)
            if c.index_dim<=0:raise ValueError('Row width must be positive')
            self.compiled=build(rows,c.index_dim,str(c.dtype).split('.')[-1])
            q,scale=self.compiled(x.reshape(rows,c.index_dim))
            return scale.reshape(shape[:-1]),q.reshape(shape)
    register_kernel_type('FP8QuantFwdOp','ascend',AscendFP8QuantKernel)
