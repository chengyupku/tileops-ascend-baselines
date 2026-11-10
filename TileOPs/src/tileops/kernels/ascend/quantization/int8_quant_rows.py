"""Ascend row/block INT8 quantization with explicit FP32 scales."""

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
    from tileops.kernels.quantization.call_spec import INT8QuantPerChannelFwdInterface,INT8QuantPerBlockFwdInterface

    @functools.lru_cache(maxsize=256)
    def build(m,k,dtype,blocked):
        group_width=128 if blocked else k
        groups_per_row=(k+group_width-1)//group_width
        groups=m*groups_per_row
        width=min(1024,tilelang.next_power_of_2(group_width))
        pitch=max(16,width)
        blocks=min(64,groups)
        @T.prim_func
        def main(X:T.Tensor((m,k),dtype),Q:T.Tensor((m,k),'int8'),S:T.Tensor((m,groups_per_row),'float32')):
            with T.Kernel(blocks) as bx:
                x=T.alloc_shared((pitch,),dtype)
                q=T.alloc_shared((pitch,),'int8')
                stats=T.alloc_shared((16,),'float32')
                bits=T.alloc_shared((16,),'int32')
                for step in T.serial((groups+blocks-1)//blocks):
                    job=step*blocks+bx
                    if job<groups:
                        row=job//groups_per_row
                        group=job%groups_per_row
                        start=group*group_width
                        valid_group=T.min(group_width,k-start)
                        T.clear(stats);T.clear(bits)
                        for chunk in T.serial((group_width+width-1)//width):
                            offset=chunk*width
                            valid=T.min(width,valid_group-offset)
                            T.copy(X[row,start+offset:start+offset+valid],x[:valid])
                            with T.SimtVF(threads=min(256,width)):
                                peak=T.alloc_reducer((1,),'int32',op='max')
                                T.reducer_init(peak)
                                for i in T.Parallel(width):
                                    if i<valid:T.reducer_update(peak[0],T.reinterpret(T.cast(x[i],'float32'),'int32') & 0x7fffffff)
                                result=T.alloc_fragment((1,),'int32')
                                T.finalize_reducer(peak,result)
                                bits[0]=T.max(bits[0],result[0])
                        with T.SimtVF(threads=32):
                            exponent=bits[0] >> 23
                            bits[1]=T.if_then_else(bits[0]>0 and exponent<=7,1,0)
                            bits[2]=0
                            if bits[1]>0:
                                mantissa=T.cast(bits[0] & 0x7fffff,'uint32') | T.if_then_else(exponent>0,T.uint32(0x800000),T.uint32(0))
                                units=mantissa << T.max(exponent-1,0)
                                quotient=units // T.uint32(127)
                                remainder=units % T.uint32(127)
                                bits[2]=T.cast(quotient+T.cast(remainder>T.uint32(63),'uint32'),'int32')
                                stats[0]=T.reinterpret(bits[2],'float32')
                            else:
                                peak_value=T.reinterpret(bits[0],'float32')
                                estimate=peak_value/T.float32(127)
                                residual=T.call_extern('float32','fmaf',-estimate,T.float32(127),peak_value)
                                corrected=estimate+residual*T.float32(1.0/127.0)
                                stats[0]=T.if_then_else(bits[0]>0,corrected,T.float32(1))
                        T.copy(stats[:1],S[row,group:group+1])
                        for chunk in T.serial((group_width+width-1)//width):
                            offset=chunk*width
                            valid=T.min(width,valid_group-offset)
                            T.copy(X[row,start+offset:start+offset+valid],x[:valid])
                            with T.SimtVF(threads=width):
                                for i in T.Parallel(width):
                                    if i<valid:
                                        value=T.cast(x[i],'float32')
                                        raw=T.reinterpret(value,'int32')
                                        magnitude=raw & 0x7fffffff
                                        exponent=magnitude >> 23
                                        mantissa=T.cast(magnitude & 0x7fffff,'uint32') | T.if_then_else(exponent>0,T.uint32(0x800000),T.uint32(0))
                                        units=mantissa << T.min(T.max(exponent-1,0),6)
                                        numerator=T.cast(units,'float32')
                                        denominator=T.cast(T.max(bits[2],1),'float32')
                                        tiny_quotient=numerator/denominator
                                        tiny_remainder=T.call_extern('float32','fmaf',-tiny_quotient,denominator,numerator)
                                        tiny_scaled=T.if_then_else(bits[2]>0,tiny_quotient+tiny_remainder/denominator,T.if_then_else(magnitude==0,T.float32(0),T.float32(127)))
                                        signed_tiny=T.if_then_else(raw<0,-tiny_scaled,tiny_scaled)
                                        normal_quotient=value/stats[0]
                                        normal_remainder=T.call_extern('float32','fmaf',-normal_quotient,stats[0],value)
                                        scaled_value=T.if_then_else(bits[1]>0,signed_tiny,normal_quotient+normal_remainder/stats[0])
                                        rounded=T.nearbyint(scaled_value)
                                        q[i]=T.cast(T.min(T.max(rounded,T.float32(-127)),T.float32(127)),'int8')
                            T.copy(q[:valid],Q[row,start+offset:start+offset+valid])
        return tilelang.compile(main,target='ascend',out_idx=[1,2])

    class _QuantRows(Kernel):
        devices:ClassVar[frozenset[str]]=frozenset({'npu'})
        @classmethod
        def entry_for(cls,call):return call,lambda:cls(call)
        def __init__(self,call):
            super().__init__();self.call=call
            if call.dtype not in (torch.float16,torch.bfloat16,torch.float32):raise TypeError('Unsupported quantization dtype')
        def forward(self,x):
            c=self.call
            if tuple(x.shape)!=(c.rows,c.cols) or x.dtype!=c.dtype:raise ValueError('Invalid quantization input signature')
            if c.rows==0:
                scale_shape=(0,(c.cols+127)//128) if self.blocked else (0,)
                return torch.empty_like(x,dtype=torch.int8),torch.empty(scale_shape,device=x.device,dtype=torch.float32)
            if c.cols<=0:raise ValueError('Quantization requires nonempty groups')
            self.compiled=build(c.rows,c.cols,str(c.dtype).split('.')[-1],self.blocked)
            q,scale=self.compiled(x)
            return q,scale if self.blocked else scale.reshape(c.rows)
    class AscendINT8QuantPerChannelKernel(_QuantRows,INT8QuantPerChannelFwdInterface):
        blocked=False
    class AscendINT8QuantPerBlockKernel(_QuantRows,INT8QuantPerBlockFwdInterface):
        blocked=True
    register_kernel_type('INT8QuantPerChannelFwdOp','ascend',AscendINT8QuantPerChannelKernel)
    register_kernel_type('INT8QuantPerBlockFwdOp','ascend',AscendINT8QuantPerBlockKernel)
