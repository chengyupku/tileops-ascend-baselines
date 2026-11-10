"""Ascend 128x128 FP8 block quantization with whole-tile buffers."""

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
    from tileops.kernels.quantization.call_spec import FP8QuantPerBlockFwdInterface

    @functools.lru_cache(maxsize=256)
    def build(m,n,dtype):
        mr=(m+127)//128;nc=(n+127)//128;jobs=mr*nc;blocks=min(64,jobs)
        @T.prim_func
        def main(X:T.Tensor((m,n),dtype),Q:T.Tensor((m,n),'float8_e4m3fn'),S:T.Tensor((mr,nc),'float32')):
            with T.Kernel(blocks) as bx:
                x=T.alloc_shared((128,128),dtype)
                q=T.alloc_shared((128,128),'float8_e4m3fn')
                stats=T.alloc_shared((16,),'float32')
                bits=T.alloc_shared((16,),'int32')
                for step in T.serial((jobs+blocks-1)//blocks):
                    job=step*blocks+bx
                    if job<jobs:
                        tr=job//nc;tc=job%nc
                        valid_rows=T.min(128,m-tr*128)
                        valid_cols=T.min(128,n-tc*128)
                        T.clear(stats);T.clear(bits);T.clear(x)
                        T.copy(X[tr*128:tr*128+valid_rows,tc*128:tc*128+valid_cols],x[:valid_rows,:valid_cols])
                        for row_group in T.serial(16):
                            with T.SimtVF(threads=256):
                                peak=T.alloc_reducer((1,),'int32',op='max')
                                nan_flag=T.alloc_reducer((1,),'int32',op='max')
                                T.reducer_init(peak);T.reducer_init(nan_flag)
                                for rr,i in T.Parallel(8,128):
                                    r=row_group*8+rr
                                    if r<valid_rows and i<valid_cols:
                                        magnitude=T.reinterpret(T.cast(x[r,i],'float32'),'int32') & 0x7fffffff
                                        T.reducer_update(peak[0],T.if_then_else(magnitude>0x7f800000,0,magnitude))
                                        T.reducer_update(nan_flag[0],T.if_then_else(magnitude>0x7f800000,1,0))
                                maximum=T.alloc_fragment((1,),'int32')
                                has_nan=T.alloc_fragment((1,),'int32')
                                T.finalize_reducer(peak,maximum);T.finalize_reducer(nan_flag,has_nan)
                                bits[0]=T.max(bits[0],maximum[0]);bits[1]=T.max(bits[1],has_nan[0])
                        with T.SimtVF(threads=32):
                            # Tiny scale is rounded as an integer multiple of 2**-149.
                            exponent=bits[0] >> 23
                            bits[2]=T.if_then_else(bits[1]==0 and bits[0]>0 and exponent<=9,1,0)
                            if bits[2]>0:
                                mantissa=T.cast(bits[0] & 0x7fffff,'uint32') | T.if_then_else(exponent>0,T.uint32(0x800000),T.uint32(0))
                                units=mantissa << T.max(exponent-1,0)
                                quotient=units // T.uint32(448)
                                remainder=units % T.uint32(448)
                                rounded=quotient+T.cast(remainder>T.uint32(224) or (remainder==T.uint32(224) and (quotient & T.uint32(1))!=0),'uint32')
                                bits[3]=T.cast(rounded,'int32')
                                stats[0]=T.reinterpret(bits[3],'float32')
                            else:
                                peak_value=T.reinterpret(bits[0],'float32')
                                estimate=peak_value/T.float32(448)
                                remainder=T.call_extern('float32','fmaf',-estimate,T.float32(448),peak_value)
                                corrected=estimate+remainder/T.float32(448)
                                scale=T.if_then_else(bits[0]==0x7f800000,peak_value,corrected)
                                stats[0]=T.if_then_else(bits[1]>0,T.float32(1),T.if_then_else(bits[0]>0,scale,T.float32(1)))
                        T.copy(stats[:1],S[tr,tc:tc+1])
                        for row_group in T.serial(16):
                            with T.SimtVF(threads=256):
                                for rr,i in T.Parallel(8,128):
                                    r=row_group*8+rr
                                    value=T.cast(x[r,i],'float32')
                                    raw=T.reinterpret(value,'int32')
                                    magnitude=raw & 0x7fffffff
                                    if bits[2]>0:
                                        exponent=magnitude >> 23
                                        mantissa=T.cast(magnitude & 0x7fffff,'uint32') | T.if_then_else(exponent>0,T.uint32(0x800000),T.uint32(0))
                                        units=mantissa << T.max(exponent-1,0)
                                        numerator=T.cast(units,'float32')
                                        denominator=T.cast(T.max(bits[3],1),'float32')
                                        quotient=numerator/denominator
                                        remainder=T.call_extern('float32','fmaf',-quotient,denominator,numerator)
                                        tiny_magnitude=T.if_then_else(bits[3]>0,quotient+remainder/denominator,T.if_then_else(magnitude==0,T.float32(float('nan')),T.float32(float('inf'))))
                                        scaled=T.if_then_else(raw<0,-tiny_magnitude,tiny_magnitude)
                                        bounded=T.if_then_else(scaled<T.float32(-448),T.float32(-448),T.if_then_else(scaled>T.float32(448),T.float32(448),scaled))
                                        # CPU reference encodes the invalid Inf/Inf result as negative NaN.
                                        if (bits[2]>0 and bits[3]==0 and magnitude==0) or (magnitude==0x7f800000 and bits[0]==0x7f800000 and bits[1]==0):
                                            q[r,i]=T.reinterpret(T.cast(255,'uint8'),'float8_e4m3fn')
                                        elif magnitude==0 and bits[1]==0:
                                            q[r,i]=T.reinterpret(T.cast(T.if_then_else(raw<0,128,0),'uint8'),'float8_e4m3fn')
                                        else:
                                            q[r,i]=T.cast(bounded,'float8_e4m3fn')
                                    else:
                                        quotient=value/stats[0]
                                        remainder=T.call_extern('float32','fmaf',-quotient,stats[0],value)
                                        scaled=T.if_then_else(magnitude<0x7f800000 and bits[0]<0x7f800000 and stats[0]>0,quotient+remainder/stats[0],quotient)
                                        bounded=T.if_then_else(scaled<T.float32(-448),T.float32(-448),T.if_then_else(scaled>T.float32(448),T.float32(448),scaled))
                                        # CPU reference encodes the invalid Inf/Inf result as negative NaN.
                                        if (bits[2]>0 and bits[3]==0 and magnitude==0) or (magnitude==0x7f800000 and bits[0]==0x7f800000 and bits[1]==0):
                                            q[r,i]=T.reinterpret(T.cast(255,'uint8'),'float8_e4m3fn')
                                        elif magnitude==0 and bits[1]==0:
                                            q[r,i]=T.reinterpret(T.cast(T.if_then_else(raw<0,128,0),'uint8'),'float8_e4m3fn')
                                        else:
                                            q[r,i]=T.cast(bounded,'float8_e4m3fn')
                        T.copy(q[:valid_rows,:valid_cols],Q[tr*128:tr*128+valid_rows,tc*128:tc*128+valid_cols])
        return tilelang.compile(main,target='ascend',out_idx=[1,2])

    class AscendFP8QuantPerBlockKernel(Kernel,FP8QuantPerBlockFwdInterface):
        devices:ClassVar[frozenset[str]]=frozenset({'npu'})
        @classmethod
        def entry_for(cls,call):return call,lambda:cls(call)
        def __init__(self,call):super().__init__();self.call=call
        def forward(self,x):
            c=self.call
            if tuple(x.shape)!=(c.rows,c.cols) or x.dtype!=c.dtype:raise ValueError('Invalid FP8 block input')
            if not x.numel():return torch.empty_like(x,dtype=torch.float8_e4m3fn),torch.empty(((c.rows+127)//128,(c.cols+127)//128),device=x.device,dtype=torch.float32)
            self.compiled=build(c.rows,c.cols,str(c.dtype).split('.')[-1])
            return self.compiled(x)
    register_kernel_type('FP8QuantPerBlockFwdOp','ascend',AscendFP8QuantPerBlockKernel)
