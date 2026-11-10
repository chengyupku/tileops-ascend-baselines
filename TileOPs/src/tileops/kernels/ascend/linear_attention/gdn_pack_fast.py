"""GDN chunk packing, fused head mapping, transforms and local decay prefixes."""
import tilelang.ascend.language as T

def pack(B,S,H,HV,K,V,dtype,scale,initial,norm,raw_gate,beta_sigmoid,negative,C=32):
    N=(S+C-1)//C;J=B*N*HV;grid=min(64,J)
    @T.prim_func
    def main(Q:T.Tensor((B,S,H,K),dtype),KK:T.Tensor((B,S,H,K),dtype),
             VV:T.Tensor((B,S,HV,V),dtype),G:T.Tensor((B,S,HV),dtype),Beta:T.Tensor((B,S,HV),dtype),
             H0:T.Tensor((B,HV,K,V),'float32'),ALog:T.Tensor((HV,),'float32'),Bias:T.Tensor((HV,),'float32'),
             QP:T.Tensor((J,C,K),dtype),KP:T.Tensor((J,C,K),dtype),VP:T.Tensor((J,C,V),dtype),
             BP:T.Tensor((J,C),'float32'),GP:T.Tensor((J,C),'float32'),
             QG:T.Tensor((J,C,K),dtype),KG:T.Tensor((J,C,K),dtype),KE:T.Tensor((J,C,K),dtype),
             HP:T.Tensor((B*HV,K,V),dtype)):
        with T.Kernel(grid) as bx:
            q=T.alloc_shared((C,K),'float32');k=T.alloc_shared((C,K),'float32')
            qi=T.alloc_shared((C,K),dtype);ki=T.alloc_shared((C,K),dtype)
            qn=T.alloc_shared((C,K),dtype);kn=T.alloc_shared((C,K),dtype)
            qg=T.alloc_shared((C,K),dtype);kg=T.alloc_shared((C,K),dtype);ke=T.alloc_shared((C,K),dtype)
            vn=T.alloc_shared((C,V),dtype);inv=T.alloc_shared((C,2),'float32')
            gates=T.alloc_shared((C,),'float32');prefix=T.alloc_shared((C,),'float32')
            hs=T.alloc_shared((K,V),'float32');hn=T.alloc_shared((K,V),dtype)
            for step in T.serial((J+grid-1)//grid):
                j=step*grid+bx
                if j<J:
                    b=j//(N*HV);n=j//HV%N;h=j%HV;qh=h//(HV//H);valid=T.min(C,S-n*C)
                    if valid<C:
                        with T.SimtVF(threads=256):
                            for t,i in T.Parallel(C,K):qi[t,i]=0.;ki[t,i]=0.
                            for t,i in T.Parallel(C,V):vn[t,i]=0.
                    T.copy(Q[b,n*C:n*C+valid,qh,:],qi[:valid,:]);T.copy(KK[b,n*C:n*C+valid,qh,:],ki[:valid,:])
                    T.copy(VV[b,n*C:n*C+valid,h,:],vn[:valid,:]);T.copy(vn,VP[j,:,:])
                    with T.SimtVF(threads=256):
                        for t,i in T.Parallel(C,K):q[t,i]=qi[t,i];k[t,i]=ki[t,i]
                        for t in T.Parallel(C):
                            if n*C+t<S:
                                gg=T.cast(G[b,n*C+t,h],'float32');bb=T.cast(Beta[b,n*C+t,h],'float32')
                                if raw_gate:
                                    x=gg+Bias[h];gg=-T.exp(ALog[h])*T.if_then_else(x>20.,x,T.log1p(T.exp(x)))
                                if beta_sigmoid:bb=(2. if negative else 1.)/(1.+T.exp(-bb))
                                gates[t]=gg;BP[j,t]=bb
                            else:
                                gates[t]=0.;BP[j,t]=0.
                    with T.SimtVF(threads=128):
                        t=T.get_thread_binding()
                        if t<C:
                            cumulative=T.alloc_local((1,),'float32');cumulative[0]=gates[t]
                            for shift in T.unroll(5):
                                offset=1<<shift
                                other=T.call_extern('float32','asc_shfl',cumulative[0],T.max(0,t-offset),32)
                                cumulative[0]=cumulative[0]+T.if_then_else(t>=offset,other,T.float32(0))
                            prefix[t]=cumulative[0];GP[j,t]=cumulative[0]
                    if norm:
                        with T.SimtVF(threads=256):
                            rq=T.alloc_reducer((C,),'float32',op='sum');rk=T.alloc_reducer((C,),'float32',op='sum');T.reducer_init(rq);T.reducer_init(rk)
                            for t,i in T.Parallel(C,K):T.reducer_update(rq[t],q[t,i]*q[t,i]);T.reducer_update(rk[t],k[t,i]*k[t,i])
                            sq=T.alloc_fragment((C,),'float32');sk=T.alloc_fragment((C,),'float32');T.finalize_reducer(rq,sq);T.finalize_reducer(rk,sk)
                            for t in T.Parallel(C):inv[t,0]=T.rsqrt(sq[t]+1e-6);inv[t,1]=T.rsqrt(sk[t]+1e-6)
                    with T.SimtVF(threads=256):
                        for t,i in T.Parallel(C,K):
                            qq=q[t,i]*(inv[t,0] if norm else T.float32(1))*scale
                            kk=k[t,i]*(inv[t,1] if norm else T.float32(1))
                            qn[t,i]=qq;kn[t,i]=kk
                            qg[t,i]=qq*T.exp(prefix[t]);kg[t,i]=kk*T.exp(prefix[t]);ke[t,i]=kk*T.exp(prefix[C-1]-prefix[t])
                    T.copy(qn,QP[j,:,:]);T.copy(kn,KP[j,:,:]);T.copy(qg,QG[j,:,:]);T.copy(kg,KG[j,:,:]);T.copy(ke,KE[j,:,:])
                    if initial:
                        if n==0:
                            T.copy(H0[b,h,:,:],hs)
                            with T.SimtVF(threads=256):
                                for i,v in T.Parallel(K,V):hn[i,v]=hs[i,v]
                            T.copy(hn,HP[b*HV+h,:,:])
    return main
