"""Candidate one persistent SIMT region with explicit workgroup synchronization."""
import tilelang.ascend.language as T

def inverse(J,dtype,C=32):
    assert C==32;grid=min(64,J)
    @T.prim_func
    def main(Gram:T.Tensor((J,C,C),'float32'),QK:T.Tensor((J,C,C),'float32'),
             Beta:T.Tensor((J,C),'float32'),Prefix:T.Tensor((J,C),'float32'),
             B:T.Tensor((J,C,C),dtype),Att:T.Tensor((J,C,C),dtype)):
        with T.Kernel(grid) as bx:
            gram=T.alloc_shared((C,C),'float32');beta=T.alloc_shared((C,),'float32')
            with T.SimtVF(threads=1024):
                tid=T.get_thread_binding();i=tid//32;j=tid%32
                value=T.alloc_local((1,),'float32')
                for step in T.serial((J+grid-1)//grid):
                    job=step*grid+bx
                    if job<J:
                        if i==0:beta[j]=Beta[job,j]
                        if j<=i:
                            decay=T.exp(Prefix[job,i]-Prefix[job,j])
                            gram[i,j]=T.if_then_else(j<i,Gram[job,i,j]*decay,T.float32(0))
                            Att[job,i,j]=QK[job,i,j]*decay
                        else:
                            gram[i,j]=0.;Att[job,i,j]=0.
                        T.evaluate(T.call_extern('void','__syncthreads'))
                        value[0]=T.if_then_else(i==j,T.float32(1),T.if_then_else(i>j,-beta[i]*gram[i,j],T.float32(0)))
                        for pivot in T.unroll(C-1):
                            u=C-1-pivot
                            coefficient=T.call_extern('float32','asc_shfl',value[0],u,32)*beta[u]
                            value[0]=T.if_then_else(i>u and j<u,value[0]-coefficient*gram[u,j],value[0])
                        B[job,i,j]=value[0]*beta[j]/T.if_then_else(beta[i]!=0.,beta[i],T.float32(1))
                        T.evaluate(T.call_extern('void','__syncthreads'))
    return main
