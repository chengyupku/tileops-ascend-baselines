"""Chunk products with a native-dtype vector epilogue kept on chip."""
import functools
import tilelang
import tilelang.ascend.language as T

def build(J,M,N,D,dtype,ta=False,mode=0,save_full=False):
    return _build(M,N,D,dtype,ta,mode,save_full)

@functools.lru_cache(maxsize=128)
def _build(M,N,D,dtype,ta,mode,save_full):
    J=T.dynamic("jobs")
    ashape=(J,D,M) if ta else (J,M,D)
    xshape=(J,M,N) if mode==2 else (J,D,N)
    fullshape=(J,M,N) if save_full else (1,)
    R=M//2;grid=32
    @T.prim_func
    def main(A:T.Tensor(ashape,dtype),B:T.Tensor((J,D,N),dtype),X:T.Tensor(xshape,dtype),
             O:T.Tensor((J,M,N),dtype),Full:T.Tensor(fullshape,'float32')):
        with T.MixedKernel(grid) as (bx,sid):
            a=T.alloc_l1((M,D),dtype);b=T.alloc_l1((N,D),dtype)
            aa=T.alloc_l0a((M,D),dtype);bb=T.alloc_l0b((N,D),dtype)
            cc=T.alloc_l0c((M,N),'float32')
            partial=T.alloc_shared((R,N),'float32')
            result=T.alloc_shared((R,N),dtype);x=T.alloc_shared((R,N),dtype)
            for step in T.Pipelined((J+grid-1)//grid,num_stages=2):
                j=step*grid+bx
                if j<J:
                    if ta:T.copy(A[j,:,:],a,transpose=True)
                    else:T.copy(A[j,:,:],a)
                    T.copy(B[j,:,:],b,transpose=True)
                    T.copy(a,aa);T.copy(b,bb)
                    T.gemm(aa,bb,cc,transpose_B=True,clear_accum=True)
                    T.dual_copy(cc,partial)
                    if mode==2:T.dual_copy(X[j,:,:],x)
                    with T.SimtVF(threads=256):
                        for i,k in T.Parallel(R,N):
                            if mode==1:result[i,k]=T.if_then_else(i+sid*R==k,T.float32(1),T.float32(0))-partial[i,k]
                            elif mode==2:result[i,k]=T.cast(x[i,k],'float32')-partial[i,k]
                            else:result[i,k]=partial[i,k]
                    T.dual_copy(result,O[j,:,:])
                    if save_full:T.dual_copy(partial,Full[j,:,:])
    return tilelang.compile(main,target='ascend',out_idx=[3,4])
