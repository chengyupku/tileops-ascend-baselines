"""Tensor-wide INT8 quantization with a bounded hierarchical maximum reduction."""

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
    from tileops.kernels.quantization.call_spec import INT8QuantPerTensorFwdInterface

    @functools.lru_cache(maxsize=256)
    def build_peak(n,dtype):
        width=4096
        chunks=(n+width-1)//width
        blocks=min(64,chunks)
        @T.prim_func
        def main(X:T.Tensor((n,),dtype),P:T.Tensor((chunks,),'float32')):
            with T.Kernel(blocks) as bx:
                x=T.alloc_shared((width,),dtype)
                p=T.alloc_shared((16,),'float32')
                for step in T.serial((chunks+blocks-1)//blocks):
                    chunk=step*blocks+bx
                    if chunk<chunks:
                        begin=chunk*width
                        valid=T.min(width,n-begin)
                        T.copy(X[begin:begin+valid],x[:valid])
                        with T.SimtVF(threads=256):
                            peak=T.alloc_reducer((1,),'int32',op='max')
                            T.reducer_init(peak)
                            for i in T.Parallel(width):
                                if i<valid:T.reducer_update(peak[0],T.reinterpret(T.cast(x[i],'float32'),'int32') & 0x7fffffff)
                            reduced=T.alloc_fragment((1,),'int32')
                            T.finalize_reducer(peak,reduced)
                            p[0]=T.reinterpret(reduced[0],'float32')
                        T.copy(p[:1],P[chunk:chunk+1])
        return tilelang.compile(main,target='ascend',out_idx=[1]),chunks

    @functools.lru_cache(maxsize=256)
    def build_output(n,dtype):
        width=1024
        chunks=(n+width-1)//width
        blocks=min(64,chunks)
        @T.prim_func
        def main(X:T.Tensor((n,),dtype),P:T.Tensor((1,),'float32'),Q:T.Tensor((n,),'int8'),S:T.Tensor((1,),'float32')):
            with T.Kernel(blocks) as bx:
                x=T.alloc_shared((width,),dtype)
                q=T.alloc_shared((width,),'int8')
                T.annotate_buffer_versions({x:2,q:2})
                s=T.alloc_shared((16,),'float32')
                bits=T.alloc_shared((16,),'int32')
                T.copy(P[:1],s[:1])
                with T.SimtVF(threads=32):
                    bits[0]=T.reinterpret(s[0],'int32')
                    exponent=bits[0] >> 23
                    bits[1]=T.if_then_else(bits[0]>0 and exponent<=7,1,0)
                    bits[2]=0
                    if bits[1]>0:
                        mantissa=T.cast(bits[0] & 0x7fffff,'uint32') | T.if_then_else(exponent>0,T.uint32(0x800000),T.uint32(0))
                        units=mantissa << T.max(exponent-1,0)
                        quotient=units // T.uint32(127)
                        remainder=units % T.uint32(127)
                        bits[2]=T.cast(quotient+T.cast(remainder>T.uint32(63),'uint32'),'int32')
                        s[0]=T.reinterpret(bits[2],'float32')
                    else:
                        peak_value=T.reinterpret(bits[0],'float32')
                        estimate=peak_value/T.float32(127)
                        residual=T.call_extern('float32','fmaf',-estimate,T.float32(127),peak_value)
                        corrected=estimate+residual*T.float32(1.0/127.0)
                        s[0]=T.if_then_else(bits[0]>0,corrected,T.float32(1))
                if bx==0:T.copy(s[:1],S[:1])
                for step in T.Pipelined((chunks+blocks-1)//blocks,num_stages=2):
                    chunk=step*blocks+bx
                    if chunk<chunks:
                        begin=chunk*width
                        valid=T.min(width,n-begin)
                        T.copy(X[begin:begin+valid],x[:valid])
                        with T.SimtVF(threads=width):
                            if bits[1]>0:
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
                                        rounded=T.nearbyint(signed_tiny)
                                        q[i]=T.cast(T.min(T.max(rounded,T.float32(-127)),T.float32(127)),'int8')
                            elif bits[0]>0 and (bits[0] >> 23)<=64:
                                for i in T.Parallel(width):
                                    if i<valid:
                                        value=T.cast(x[i],'float32')
                                        raw=T.reinterpret(value,'int32')
                                        magnitude=raw & 0x7fffffff
                                        exponent=magnitude >> 23
                                        expanded_magnitude=T.if_then_else(exponent>0,T.reinterpret(magnitude+(64 << 23),'float32'),T.cast(magnitude,'float32')*T.float32(2.0**-85))
                                        expanded=T.if_then_else(raw<0,-expanded_magnitude,expanded_magnitude)
                                        expanded_scale=T.reinterpret(T.reinterpret(s[0],'int32')+(64 << 23),'float32')
                                        quotient=expanded/expanded_scale
                                        remainder=T.call_extern('float32','fmaf',-quotient,expanded_scale,expanded)
                                        rounded=T.nearbyint(quotient+remainder/expanded_scale)
                                        q[i]=T.cast(T.min(T.max(rounded,T.float32(-127)),T.float32(127)),'int8')
                            else:
                                for i in T.Parallel(width):
                                    if i<valid:
                                        value=T.cast(x[i],'float32')
                                        normal_quotient=value/s[0]
                                        normal_remainder=T.call_extern('float32','fmaf',-normal_quotient,s[0],value)
                                        rounded=T.nearbyint(normal_quotient+normal_remainder/s[0])
                                        q[i]=T.cast(T.min(T.max(rounded,T.float32(-127)),T.float32(127)),'int8')
                        T.copy(q[:valid],Q[begin:begin+valid])
        return tilelang.compile(main,target='ascend',out_idx=[2,3])

    class AscendINT8QuantPerTensorKernel(Kernel,INT8QuantPerTensorFwdInterface):
        devices:ClassVar[frozenset[str]]=frozenset({'npu'})
        @classmethod
        def entry_for(cls,call):return call,lambda:cls(call)
        def __init__(self,call):
            super().__init__();self.call=call
            if call.dtype not in (torch.float16,torch.bfloat16,torch.float32):raise TypeError('Unsupported quantization dtype')
        def forward(self,x):
            c=self.call
            if tuple(x.shape)!=(c.rows,c.cols) or x.dtype!=c.dtype:raise ValueError('Invalid quantization input signature')
            if not x.numel():raise ValueError('Quantization requires nonempty input')
            flat=x.reshape(-1)
            peak=flat
            dtype=str(c.dtype).split('.')[-1]
            self.reductions=[]
            while True:
                kernel,count=build_peak(peak.numel(),dtype)
                self.reductions.append(kernel)
                peak=kernel(peak)
                if count==1:break
                dtype='float32'
            self.compiled=build_output(flat.numel(),str(c.dtype).split('.')[-1])
            q,scale=self.compiled(flat,peak)
            return q.reshape_as(x),scale

    register_kernel_type('INT8QuantPerTensorFwdOp','ascend',AscendINT8QuantPerTensorKernel)
