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
mod=load(P.parent/'bf16_moe_hadamard_parent10/adapter.py'); calls=[]
def native(a,b,*,metadata):
 assert a.dtype==b.dtype=='fp32' and a.device==b.device and a.shape[1]==b.shape[0]
 calls.append((a,b,metadata));return T((a.shape[0],b.shape[1]),'fp32',a.device)
b=mod.make_binding(native)
y=b.moe(T((32,128,7168)),T((32,4096,7168)),Meta([96]*32),masked=True)
assert y.shape==(32,128,4096) and len(calls)==32 and len(y.writes)==32
assert sum(c[2]['count'] for c in calls)==3072
assert all(c[0].shape==(96,7168) and c[1].shape==(7168,4096) for c in calls)
calls.clear()
for counts in ([129]+[96]*31,[-1]+[96]*31,[96.0]*32):
 try:b.moe(T((32,128,7168)),T((32,4096,7168)),Meta(counts),masked=True)
 except ValueError:pass
 else:raise AssertionError('invalid metadata accepted')
 assert not calls
y=b.moe(T((2,4,3)),T((2,6,3)),Meta([1,0]),masked=True,out_dtype='fp32')
assert len(calls)==1 and y.dtype=='fp32' and y.shape==(2,4,6)
del sys.modules['torch']
result=dict(status='pass',mode='stdlib strict torch-compatible shape/slice mock from parent10; not numerical validation',checks=['32 experts ENK transpose FP32 product','3072 defined rows full output scatter','invalid counts rejected before native','empty expert skipped requested output dtype'],numeric_validation=False)
(P/'CPU_CHECKS.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
