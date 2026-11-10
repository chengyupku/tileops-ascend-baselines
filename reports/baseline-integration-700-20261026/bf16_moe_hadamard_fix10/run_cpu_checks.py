import json, sys, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

class DType:
    def __init__(self,name): self.name=name
    def __repr__(self): return self.name
BF16=DType('bfloat16'); FP32=DType('float32')
class T:
    __module__='strict_fake'
    def __init__(self,shape,device='npu:0',dtype=BF16,tag='tensor',value=None): self.shape=tuple(shape); self.device=device; self.dtype=dtype; self.tag=tag; self.value=value
    def __getitem__(self,k):
        if not isinstance(k,tuple): k=(k,)
        out=[]
        for i,d in enumerate(self.shape):
            q=k[i] if i<len(k) else slice(None)
            if isinstance(q,int): continue
            st=0 if q.start is None else q.start; en=d if q.stop is None else q.stop; out.append(en-st)
        return T(out,self.device,self.dtype,'slice')
    def __setitem__(self,k,v): self.last_set=(k,v)
    def float(self): return T(self.shape,self.device,FP32,'float')
    def to(self,dtype): return T(self.shape,self.device,dtype,'to')
    def contiguous(self): return T(self.shape,self.device,self.dtype,'contiguous',self.value)
    def reshape(self,*s):
        if len(s)==1 and isinstance(s[0],tuple): s=s[0]
        return T(s,self.device,self.dtype,'reshape',self.value)
    def transpose(self,a,b):
        if not (isinstance(a,int) and isinstance(b,int)): raise TypeError('two dims required')
        sh=list(self.shape); sh[a],sh[b]=sh[b],sh[a]; return T(sh,self.device,self.dtype,'transpose',self.value)
    def __repr__(self): return f'T{self.shape}@{self.device}/{self.dtype}'

def zeros(shape,dtype,device): return T(shape,device,dtype,'zeros')
def tensor(rows,dtype,device): return T((len(rows),len(rows[0]) if rows else 0),device,dtype,'coeff',rows)
sys.modules['torch']=types.SimpleNamespace(zeros=zeros,tensor=tensor,float32=FP32)
from adapter import make_binding
calls=[]
def native(lhs,rhs,*,metadata):
    calls.append((lhs,rhs,metadata)); return T((lhs.shape[0],rhs.shape[1]),lhs.device,FP32,'native')
b=make_binding(native); checks=[]
# cumulative physical ends [2,2,3], E,N,K and non-square dimensions
A=T((3,5)); B=T((3,7,5)); y=b.moe(A,B,T((3,),dtype=FP32),out_dtype=FP32) if False else b.moe(A,B,types.SimpleNamespace(tolist=lambda:[2,2,3]),out_dtype=FP32)
assert y.shape==(3,7) and y.device=='npu:0' and [c[2].get('start') for c in calls]==[0,2]
assert calls[0][1].shape==(5,7) and calls[0][0].dtype is FP32; checks.append({'name':'physical_psum_end_and_weight_transpose','pass':True})
# masked counts, empty skipped, full shape/device preserved
calls.clear(); y=b.moe(T((2,4,3)),T((2,6,3)),types.SimpleNamespace(tolist=lambda:[1,0]),masked=True,out_dtype=FP32)
assert y.shape==(2,4,6) and y.device=='npu:0' and len(calls)==1; checks.append({'name':'masked_counts_fullshape_empty','pass':True})
# n128 two calls and FP32 then final cast
calls.clear(); y=b.hadamard(T((2,128)),n=128); assert len(calls)==2 and calls[0][0].dtype is FP32 and calls[1][2]['k']==1 and y.dtype is BF16; checks.append({'name':'hadamard_n128_two_stage_cast','pass':True})
# n4096 two-stage and coefficient values/order, 2-d transpose signature
calls.clear(); y=b.hadamard(T((1,4096)),n=4096); assert len(calls)==2 and calls[1][2]['normalization']==1/64 and calls[0][1].value[0][1]==1.0 and calls[1][1].value[1][1]==-1/64; checks.append({'name':'hadamard_n4096_coefficients','pass':True})
print(json.dumps({'status':'pass','checks':checks,'numerical_validation':False},indent=2))
