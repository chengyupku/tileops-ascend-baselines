import sys,types,math,json,pathlib,importlib.util,hashlib
P=pathlib.Path(__file__).resolve().parent
class T:
 def __init__(self,shape,dtype='bf16',device='npu:0',values=None,trace=()):self.shape=tuple(shape);self.dtype=dtype;self.device=device;self.values=values;self.trace=trace;self.writes=[]
 def new(self,shape,action,dtype=None,values=None):return T(shape,dtype or self.dtype,self.device,self.values if values is None else values,self.trace+(action,))
 def float(self):return self.new(self.shape,('cast','fp32'),'fp32')
 def to(self,dtype):return self.new(self.shape,('cast',dtype),dtype)
 def transpose(self,a,b):
  dims=list(self.shape);dims[a],dims[b]=dims[b],dims[a];return self.new(dims,('transpose',a,b))
 def contiguous(self):return self.new(self.shape,('contiguous',))
 def reshape(self,*shape):
  if len(shape)==1 and isinstance(shape[0],tuple):shape=shape[0]
  assert math.prod(shape)==math.prod(self.shape)
  return self.new(shape,('reshape',shape))
 def __getitem__(self,key):
  if not isinstance(key,tuple):key=(key,)
  shape=[]
  for i,n in enumerate(self.shape):
   k=key[i] if i<len(key) else slice(None)
   if isinstance(k,int):assert 0<=k<n
   else:
    start,stop,step=k.indices(n);assert step==1;shape.append(max(0,stop-start))
  return self.new(shape,('slice',key))
 def __setitem__(self,key,v):
  assert self[key].shape==v.shape and v.dtype==self.dtype and v.device==self.device
  self.writes.append((key,v))
class Meta:
 def __init__(self,v):self.v=v
 def tolist(self):return self.v[:]
fake=types.SimpleNamespace(float32='fp32',zeros=lambda shape,*,dtype,device:T(shape,dtype,device),tensor=lambda rows,*,dtype,device:T((len(rows),len(rows[0])),dtype,device,rows))
assert 'torch' not in sys.modules;sys.modules['torch']=fake

def load(p):
 spec=importlib.util.spec_from_file_location('review_'+hashlib.sha256(str(p).encode()).hexdigest()[:8],p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
mod=load(P/'adapter.py');old=load(P.parent/'bf16_moe_hadamard_fix10/adapter.py');calls=[]
def native(a,b,*,metadata):
 assert a.dtype==b.dtype=='fp32' and a.device==b.device and a.shape[1]==b.shape[0]
 calls.append((a,b,metadata));return T((a.shape[0],b.shape[1]),'fp32',a.device,trace=(('native',len(calls)),))
b=mod.make_binding(native);tests=[]
def reject(fn):
 try:fn()
 except ValueError:return
 raise AssertionError('expected rejection')
y=b.moe(T((3,5)),T((3,7,5)),Meta([2,2,3]),out_dtype='fp32')
assert y.shape==(3,7) and y.dtype=='fp32' and y.device=='npu:0'
assert [(x[2]['expert'],x[2]['start'],x[2]['end']) for x in calls]==[(0,0,2),(2,2,3)]
assert [(k[0].start,k[0].stop) for k,v in y.writes]==[(0,2),(2,3)]
assert all(v.dtype=='fp32' and ('cast','bf16') not in v.trace for _,v in y.writes)
tests.append('physical_psum_empty_middle_ENK_scatter_FP32')
for bad in [[2,1,3],[-1,2,3],[2,2,4],[1,1,2],[2.0,2,3]]:
 calls.clear();reject(lambda:b.moe(T((3,5)),T((3,7,5)),Meta(bad)));assert not calls
# Prove old purported fix did not reject decreasing ends BEFORE any native calls.
calls.clear()
try:old.make_binding(native).moe(T((3,5)),T((3,7,5)),Meta([2,1,3]))
except Exception:pass
assert calls;tests.append('reject_nonmonotonic_before_native_old_bug_reproduced')
calls.clear();y=b.moe(T((2,4,3)),T((2,6,3)),Meta([1,0]),masked=True)
assert y.shape==(2,4,6) and y.dtype=='bf16' and y.device=='npu:0' and len(calls)==1
assert y.writes[0][0][0]==0 and y.writes[0][0][1].stop==1
calls.clear();reject(lambda:b.moe(T((2,4,3)),T((2,6,3)),Meta([1,5]),masked=True));assert not calls
tests.append('masked_counts_fullshape_device_empty_and_rejection')
for n,order in [(128,1),(4096,64)]:
 calls.clear();y=b.hadamard(T((2,n)));assert y.shape==(2,n) and y.dtype=='bf16' and y.device=='npu:0' and len(calls)==2
 radix=n//order;a0,h0,_=calls[0];a1,h1,_=calls[1]
 assert a0.shape==(2*order,radix) and h0.shape==(radix,radix)
 assert a1.shape==(2*radix,order) and h1.shape==(order,order)
 assert h0.values==[[(-1. if (i&j).bit_count()%2 else 1.) for j in range(radix)] for i in range(radix)]
 assert h1.values==[[(-1. if (i&j).bit_count()%2 else 1.)/math.sqrt(n) for j in range(order)] for i in range(order)]
 assert all(t.dtype=='fp32' and t.device=='npu:0' for call in calls for t in call[:2])
 tests.append('Hadamard_'+str(n)+'_original_two_stage_all_coefficients_shapes_FP32_device')
calls.clear();old.make_binding(native).hadamard(T((2,128)));assert calls[0][1].values[0][0]!=1.0 and calls[1][1].values==[[1.0]];tests.append('old_n128_normalization_stage_bug_reproduced')
for dtype,device in [('bf16','npu:0'),('fp32','cpu')]:
 bad=mod.make_binding(lambda a,w,metadata:T((a.shape[0],w.shape[1]),dtype,device));reject(lambda:bad.moe(T((3,5)),T((3,7,5)),Meta([2,2,3]),out_dtype='fp32'))
tests.append('reject_native_wrong_dtype_or_device_before_scatter')
del sys.modules['torch'];assert 'torch_npu' not in sys.modules
result={'status':'pass','checks':tests,'checks_count':len(tests),'mode':'stdlib strict mock and fixed coefficient comparisons; no actual torch/native/reference/NPU computation','numeric_validation':False};(P/'CPU_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
