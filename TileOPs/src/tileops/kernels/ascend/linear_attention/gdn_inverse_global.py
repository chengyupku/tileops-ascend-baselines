"""Candidate separate gate formation and persistent SIMT inverse rows."""
import tilelang
import tilelang.ascend.language as T

def gate(J,dtype,C=32):
    grid=min(64,J)
    @T.prim_func
    def main(Gram:T.Tensor((J,C,C),'float32'),QK:T.Tensor((J,C,C),'float32'),
             Prefix:T.Tensor((J,C),'float32'),Gated:T.Tensor((J,C,C),'float32'),Att:T.Tensor((J,C,C),dtype)):
        with T.Kernel(grid) as bx:
            with T.SimtVF(threads=1024):
                tid=T.get_thread_binding();i=tid//32;j=tid%32
                for step in T.serial((J+grid-1)//grid):
                    job=step*grid+bx
                    if job<J:
                        if j<=i:
                            decay=T.exp(Prefix[job,i]-Prefix[job,j])
                            Gated[job,i,j]=T.if_then_else(j<i,Gram[job,i,j]*decay,T.float32(0))
                            Att[job,i,j]=QK[job,i,j]*decay
                        else:
                            Gated[job,i,j]=0.;Att[job,i,j]=0.
    return main

def invert(J,dtype,C=32):
    grid=min(64,J)
    @T.prim_func
    def main(Gram:T.Tensor((J,C,C),'float32'),Beta:T.Tensor((J,C),'float32'),B:T.Tensor((J,C,C),dtype)):
        with T.Kernel(grid) as bx:
            with T.SimtVF(threads=1024):
                tid=T.get_thread_binding();i=tid//32;j=tid%32
                value=T.alloc_local((1,),'float32')
                for step in T.serial((J+grid-1)//grid):
                    job=step*grid+bx
                    if job<J:
                        value[0]=T.if_then_else(i==j,T.float32(1),T.if_then_else(i>j,-Beta[job,i]*Gram[job,i,j],T.float32(0)))
                        for pivot in T.unroll(C-1):
                            u=C-1-pivot
                            if i>u:
                                coefficient=T.call_extern('float32','asc_shfl',value[0],u,32)*Beta[job,u]
                                value[0]=T.if_then_else(j<u,value[0]-coefficient*Gram[job,u,j],value[0])
                        B[job,i,j]=value[0]*Beta[job,j]/T.if_then_else(Beta[job,i]!=0.,Beta[job,i],T.float32(1))
    return main

class Inverse:
    def __init__(self,J,dtype):
        self.gate=tilelang.compile(gate(J,dtype),target='ascend',out_idx=[3,4])
        self.invert=tilelang.compile(invert(J,dtype),target='ascend',out_idx=[2])
    def __call__(self,gram,qk,beta,prefix):
        gated,att=self.gate(gram,qk,prefix)
        return self.invert(gated,beta),att
    def get_kernel_source(self):
        return self.gate.get_kernel_source()+'\n'+self.invert.get_kernel_source()
