"""Bounded dense GDN chunk pipeline with complete affine state propagation."""
import functools
import tilelang
from .gdn_pack import pack
from .gdn_inverse import inverse
from .gdn_affine import build as affine
from .deltanet_chunked import matmul,state_inputs
from .deltanet_training_fwd import weighted_wu
from .deltanet_pairs import build as paired
from .deltanet_scan import build_scan
from .deltanet_group_scan import build_group_scan
from .deltanet_seeded_states import build as seeded_states
from .deltanet_output import output

class ChunkedGDN:
    def __init__(self,B,S,H,HV,K,V,dtype,scale,initial,norm,raw_gate,beta_sigmoid,negative):
        self.length=S;self.N=N=(S+31)//32;self.initial=initial;J=B*N*HV;C=32
        compile=lambda f,idx:tilelang.compile(f,target='ascend',out_idx=idx)
        self.pack=compile(pack(B,S,H,HV,K,V,dtype,scale,initial,norm,raw_gate,beta_sigmoid,negative),list(range(8,17)))
        self.gram=matmul(J,C,C,K,dtype,tb=True)
        self.inverse=compile(inverse(J,dtype),[4,5])
        self.wu=weighted_wu(C,K,V,dtype)
        self.affine=affine(K,V,C,dtype,save_full=N==1)
        self.qbias=paired(C,K,V,C,dtype,first_mode=2,second_fp32=True)
        self.group=build_group_scan(B,N,HV,K,V,dtype,initial,G=4) if N>=64 else None
        self.scan=build_scan(B,N,HV,K,V,dtype) if self.group is None and N>1 else None
        self.states=seeded_states(B,N,HV,K,V,dtype) if initial else compile(state_inputs(B,N,HV,K,V,dtype,False),[3,4])
        self.readout=matmul(J,C,V,K,dtype)
        self.output=compile(output(B,S,HV,V,dtype),[2])
    def __call__(self,q,k,v,g,beta,h0,A_log,dt_bias):
        qp,kp,vp,bp,gp,qg,kg,ke,hp=self.pack(q,k,v,g,beta,h0,A_log,dt_bias)
        b,att=self.inverse(self.gram(kp,kp,0),self.gram(qp,kp,0),bp,gp)
        w,u=self.wu(b,kg,vp,bp)
        m,state,full=self.affine(ke,w,u,w,gp)
        qe,bias,_=self.qbias(att,w,u,qg)
        if self.group is not None:m,full=self.group(m,state)
        elif self.scan is not None:
            stride=1
            while stride<self.N:m,state,full=self.scan(m,state,stride);stride*=2
        before,final=self.states(m,full,hp) if self.initial else self.states(full,full,h0)
        result=self.output(self.readout(qe,before,0),bias)
        return result[:,:self.length].contiguous(),final
    def get_kernel_source(self):
        return '\n'.join(x.get_kernel_source() for x in vars(self).values() if hasattr(x,'get_kernel_source'))

build=functools.lru_cache(maxsize=96)(ChunkedGDN)
