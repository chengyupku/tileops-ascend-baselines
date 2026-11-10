import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from adapter import make_binding
class Fake:
    def __init__(self, shape, tag='tensor'): self.shape=tuple(shape); self.tag=tag
    def __getitem__(self, key):
        if not isinstance(key, tuple): key=(key,)
        out=[]
        for i,d in enumerate(self.shape):
            k=key[i] if i<len(key) else slice(None)
            out.append((k.stop if isinstance(k,slice) and k.stop is not None else (1 if isinstance(k,int) else d)) - (k.start or 0) if isinstance(k,slice) else None)
        return Fake(tuple(d for d in out if d is not None), 'slice')
    def __setitem__(self,key,val): pass
    def reshape(self,*shape):
        if len(shape)==1 and isinstance(shape[0],tuple): shape=shape[0]
        return Fake(shape,'reshape')
    def transpose(self,*axes):
        if len(axes)==1 and isinstance(axes[0],tuple): axes=axes[0]
        return Fake(tuple(self.shape[i] for i in axes),'transpose')
    def zeros_like_shape(self,shape,dtype=None): return Fake(shape,'zeros')
    def matrix_like(self,shape): return Fake(shape,'matrix')
calls=[]
def mock(lhs,rhs,*,out_dtype=None,metadata=None):
    calls.append({'lhs':lhs.shape,'rhs':rhs.shape,'out_dtype':out_dtype,'metadata':metadata})
    return Fake((lhs.shape[0],rhs.shape[1]),'native')
b=make_binding(mock); checks=[]
y=b.moe(Fake((3,2)),Fake((3,2,4)),[2,0,1],out_dtype='float32'); assert y.shape==(3,4) and len(calls)==2
checks.append({'name':'moe_physical_counts_empty','pass':True,'detail':'counts are per-expert physical rows; empty expert skipped'})
calls.clear(); y=b.moe(Fake((2,3,2)),Fake((2,2,4)),[1,0],out_dtype='float32',masked=True); assert y.shape==(2,3,4) and len(calls)==1
checks.append({'name':'moe_masked_counts_padding','pass':True,'detail':'masked full shape retained and zero-count expert skipped'})
assert calls[0]['out_dtype']=='float32'; checks.append({'name':'moe_output_dtype_passthrough','pass':True,'detail':'FP32 request reaches native'})
calls.clear(); assert b.hadamard(Fake((2,128)),n=128,out_dtype='bfloat16').shape==(2,128) and len(calls)==1
checks.append({'name':'hadamard_n128_order1','pass':True,'detail':'one fixed coefficient V5 call'})
calls.clear(); assert b.hadamard(Fake((1,4096)),n=4096,out_dtype='bfloat16').shape==(1,4096) and len(calls)==2 and [c['metadata']['stage'] for c in calls]==[1,2]
assert calls[1]['metadata']['normalization']==1/64
checks.append({'name':'hadamard_n4096_two_stage','pass':True,'detail':'64x64 order and 1/64 normalization metadata preserved'})
try: make_binding(None); raise AssertionError
except TypeError: pass
checks.append({'name':'explicit_native_required','pass':True,'detail':'no fallback'})
print(json.dumps({'status':'pass','checks':checks,'numerical_validation':False},indent=2))
