"""Candidate dense GDN pipeline with sequential chunk state propagation."""
import functools
import tilelang
from .gdn_pack_fast import pack
from .gdn_inverse_global import Inverse
from .gdn_sequential_fused import build as sequential
from .deltanet_chunked import matmul
from .deltanet_training_fwd import weighted_wu
from .deltanet_pairs import build as paired
from .deltanet_output import output

class ChunkedGDNGlobal:
    def __init__(self,B,S,H,HV,K,V,dtype,scale,initial,norm,raw_gate,beta_sigmoid,negative):
        self.length=S;self.N=N=(S+31)//32;self.initial=initial;J=B*N*HV;C=32
        compile=lambda f,idx:tilelang.compile(f,target='ascend',out_idx=idx)
        self.pack=compile(pack(B,S,H,HV,K,V,dtype,scale,initial,norm,raw_gate,beta_sigmoid,negative),list(range(8,17)))
        self.gram=matmul(J,C,C,K,dtype,tb=True)
        self.inverse=Inverse(J,dtype)
        self.wu=weighted_wu(C,K,V,dtype)
        self.qbias=paired(C,K,V,C,dtype,first_mode=2,second_fp32=True)
        self.states=sequential(B,N,HV,K,V,dtype,initial)
    def __call__(self,q,k,v,g,beta,h0,A_log,dt_bias):
        qp,kp,vp,bp,gp,qg,kg,ke,hp=self.pack(q,k,v,g,beta,h0,A_log,dt_bias)
        b,att=self.inverse(self.gram(kp,kp,0),self.gram(qp,kp,0),bp,gp)
        w,u=self.wu(b,kg,vp,bp)
        qe,bias,_=self.qbias(att,w,u,qg)
        result,final=self.states(w,u,ke,gp,h0,qe,bias)
        return result[:,:self.length].contiguous(),final
    def get_kernel_source(self):
        return '\n'.join(x.get_kernel_source() for x in vars(self).values() if hasattr(x,'get_kernel_source'))

build=functools.lru_cache(maxsize=96)(ChunkedGDNGlobal)
