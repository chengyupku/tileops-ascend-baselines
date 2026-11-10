"""Candidate warp-per-token pack without intermediate UB conversion buffers."""
import tilelang.ascend.language as T

def pack(B,S,H,HV,K,V,dtype,scale,initial,norm,raw_gate,beta_sigmoid,negative,C=32):
    assert C==32 and K in (64,128) and V in (64,128)
    N=(S+C-1)//C;J=B*N*HV;grid=min(64,J);F=K//32;FV=V//32
    @T.prim_func
    def main(Q:T.Tensor((B,S,H,K),dtype),KK:T.Tensor((B,S,H,K),dtype),
             VV:T.Tensor((B,S,HV,V),dtype),G:T.Tensor((B,S,HV),dtype),Beta:T.Tensor((B,S,HV),dtype),
             H0:T.Tensor((B,HV,K,V),'float32'),ALog:T.Tensor((HV,),'float32'),Bias:T.Tensor((HV,),'float32'),
             QP:T.Tensor((J,C,K),dtype),KP:T.Tensor((J,C,K),dtype),VP:T.Tensor((J,C,V),dtype),
             BP:T.Tensor((J,C),'float32'),GP:T.Tensor((J,C),'float32'),
             QG:T.Tensor((J,C,K),dtype),KG:T.Tensor((J,C,K),dtype),KE:T.Tensor((J,C,K),dtype)):
        with T.Kernel(grid) as bx:
            with T.SimtVF(threads=1024):
                tid=T.get_thread_binding();row=tid//32;lane=tid%32
                q=T.alloc_local((F,),'float32');k=T.alloc_local((F,),'float32')
                prefix=T.alloc_local((1,),'float32');sums=T.alloc_local((2,),'float32')
                for step in T.serial((J+grid-1)//grid):
                    job=step*grid+bx
                    if job<J:
                        b=job//(N*HV);n=job//HV%N;h=job%HV;qh=h//(HV//H);token=n*C+row
                        if n*C+lane<S:
                            gg=T.cast(G[b,n*C+lane,h],'float32')
                            if raw_gate:
                                x=gg+Bias[h];gg=-T.exp(ALog[h])*T.if_then_else(x>20.,x,T.log1p(T.exp(x)))
                            prefix[0]=gg
                        else:prefix[0]=0.
                        for shift in T.unroll(5):
                            offset=1<<shift
                            other=T.call_extern('float32','asc_shfl',prefix[0],T.max(0,lane-offset),32)
                            prefix[0]=prefix[0]+T.if_then_else(lane>=offset,other,T.float32(0))
                        current=T.call_extern('float32','asc_shfl',prefix[0],row,32)
                        end=T.call_extern('float32','asc_shfl',prefix[0],31,32)
                        eg=T.exp(current);ek=T.exp(end-current)
                        if lane==0:
                            GP[job,row]=current
                            if token<S:
                                bb=T.cast(Beta[b,token,h],'float32')
                                if beta_sigmoid:bb=(2. if negative else 1.)/(1.+T.exp(-bb))
                                BP[job,row]=bb
                            else:BP[job,row]=0.
                        sums[0]=0.;sums[1]=0.
                        for f in T.unroll(F):
                            col=lane+f*32
                            if token<S:
                                q[f]=Q[b,token,qh,col];k[f]=KK[b,token,qh,col]
                            else:q[f]=0.;k[f]=0.
                            if norm:
                                sums[0]=sums[0]+q[f]*q[f];sums[1]=sums[1]+k[f]*k[f]
                        if norm:
                            for shift in T.unroll(5):
                                sums[0]=sums[0]+T.call_extern('float32','asc_shfl_xor',sums[0],1<<shift)
                                sums[1]=sums[1]+T.call_extern('float32','asc_shfl_xor',sums[1],1<<shift)
                            sums[0]=T.rsqrt(sums[0]+1e-6);sums[1]=T.rsqrt(sums[1]+1e-6)
                        for f in T.unroll(F):
                            col=lane+f*32
                            qq=q[f]*(sums[0] if norm else T.float32(1))*scale
                            kk=k[f]*(sums[1] if norm else T.float32(1))
                            QP[job,row,col]=qq;KP[job,row,col]=kk
                            QG[job,row,col]=qq*eg;KG[job,row,col]=kk*eg;KE[job,row,col]=kk*ek
                        for f in T.unroll(FV):
                            col=lane+f*32
                            if token<S:VP[job,row,col]=VV[b,token,h,col]
                            else:VP[job,row,col]=0.
    return main
