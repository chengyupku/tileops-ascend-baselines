"""Candidate fused state recurrence and readout; avoid per-chunk state GM roundtrip."""
import functools
import tilelang
import tilelang.ascend.language as T


def prim(B,N,H,K,V,dtype,initial,VT=64,C=32):
    assert V%VT==0
    J=B*N*H;tiles=V//VT;jobs=B*H*tiles;grid=min(32,jobs);R=K//2;RC=C//2
    @T.prim_func
    def main(W:T.Tensor((J,C,K),dtype),U:T.Tensor((J,C,V),dtype),KE:T.Tensor((J,C,K),dtype),
             G:T.Tensor((J,C),'float32'),H0:T.Tensor((B,H,K,V),'float32'),
             QE:T.Tensor((J,C,K),dtype),Att:T.Tensor((J,C,C),dtype),
             O:T.Tensor((B,N*C,H,V),dtype),HT:T.Tensor((B,H,K,V),'float32')):
        with T.MixedKernel(grid) as (bx,sid):
            state=T.alloc_shared((R,VT),'float32');native=T.alloc_shared((R,VT),dtype)
            combined0=T.alloc_shared((C,VT),'float32');u=T.alloc_shared((RC,VT),dtype)
            residual=T.alloc_shared((RC,VT),dtype);combined1=T.alloc_shared((RC+R,VT),'float32')
            h1=T.alloc_l1((K,VT),dtype);w1=T.alloc_l1((2*C,K),dtype)
            k1=T.alloc_l1((K+C,C),dtype);r1=T.alloc_l1((C,VT),dtype)
            wa=T.alloc_l0a((2*C,K),dtype);hb=T.alloc_l0b((K,VT),dtype)
            ka=T.alloc_l0a((K+C,C),dtype);rb=T.alloc_l0b((C,VT),dtype)
            p0=T.alloc_l0c((2*C,VT),'float32');s0=T.alloc_l0c((K+C,VT),'float32')
            for step in T.serial((jobs+grid-1)//grid):
                job=step*grid+bx
                if job<jobs:
                    b=job//(H*tiles);h=job//tiles%H;col=job%tiles*VT
                    if initial:T.dual_copy(H0[b,h,:,col:col+VT],state)
                    else:
                        with T.SimtVF(threads=512):
                            for i,v in T.Parallel(R,VT):state[i,v]=0.
                    with T.SimtVF(threads=512):
                        for i,v in T.Parallel(R,VT):native[i,v]=state[i,v]
                    for n in T.serial(N):
                        j=(b*N+n)*H+h
                        T.dual_copy(native,h1)
                        T.copy(W[j,:RC,:],w1[:RC,:]);T.copy(QE[j,:RC,:],w1[RC:C,:])
                        T.copy(W[j,RC:C,:],w1[C:C+RC,:]);T.copy(QE[j,RC:C,:],w1[C+RC:2*C,:])
                        T.copy(w1,wa);T.copy(h1,hb)
                        T.gemm(wa,hb,p0,clear_accum=True)
                        T.dual_copy(p0,combined0)
                        T.dual_copy(U[j,:,col:col+VT],u)
                        with T.SimtVF(threads=512):
                            for i,v in T.Parallel(RC,VT):residual[i,v]=T.cast(u[i,v],'float32')-combined0[i,v]
                        T.dual_copy(residual,r1);T.copy(r1,rb)
                        T.copy(Att[j,:RC,:],k1[:RC,:]);T.copy(KE[j,:,:R],k1[RC:RC+R,:],transpose=True)
                        T.copy(Att[j,RC:C,:],k1[RC+R:C+R,:]);T.copy(KE[j,:,R:K],k1[C+R:C+K,:],transpose=True)
                        T.copy(k1,ka);T.gemm(ka,rb,s0,clear_accum=True)
                        T.dual_copy(s0,combined1)
                        with T.SimtVF(threads=512):
                            for i,v in T.Parallel(RC,VT):O[b,n*C+sid*RC+i,h,col+v]=combined0[RC+i,v]+combined1[i,v]
                        with T.SimtVF(threads=512):
                            for i,v in T.Parallel(R,VT):
                                state[i,v]=T.exp(G[j,C-1])*state[i,v]+combined1[RC+i,v]
                                native[i,v]=state[i,v]
                    T.dual_copy(state,HT[b,h,:,col:col+VT])
    return main

@functools.lru_cache(maxsize=96)
def build(B,N,H,K,V,dtype,initial,VT=64):
    return tilelang.compile(prim(B,N,H,K,V,dtype,initial,VT),target='ascend',out_idx=[7,8])
