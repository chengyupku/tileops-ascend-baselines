import os,sys,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class BlockNpu:
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='torch_npu' or fullname.startswith('torch_npu.'):raise RuntimeError('CPU_ONLY')
sys.meta_path.insert(0,BlockNpu())
sys.path[:0]=['/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages','/opt/tileops-ci/runtime/lib/python3.12/site-packages']
import torch,numpy as np
P=Path(__file__).resolve().parent
meta=json.loads((P/'CAPTURE.json').read_text())
def load(name):
 raw=(P/(name+'.bin')).read_bytes();m=meta[name]
 assert hashlib.sha256(raw).hexdigest()==m['sha256']
 return torch.frombuffer(bytearray(raw),dtype=torch.bfloat16).reshape(m['shape'])
a=load('candidate_0_input0');b=load('candidate_0_input1');c=load('candidate_0_output');r=load('reference_0_output')
assert a.shape==b.shape==(1024,1024)
assert all(meta[f'candidate_0_input{i}']['sha256']==meta[f'reference_0_input{i}']['sha256']==meta[f'candidate_0_post_input{i}']['sha256']==meta[f'reference_0_post_input{i}']['sha256'] for i in range(2))
torch.set_num_threads(4)
f64=a.double()@b.double();f32=a.float()@b.float();z=f64.bfloat16();f=f32.bfloat16()
def save(name,x):
 raw=x.contiguous().view(torch.uint8).numpy().tobytes();(P/(name+'.bin')).write_bytes(raw)
 return {'sha256':hashlib.sha256(raw).hexdigest(),'shape':list(x.shape),'dtype':str(x.dtype)}
def compare(x,t,name):
 xd=x.double().numpy();td=t.double().numpy();signed=xd-td;delta=np.abs(signed);limit=.016+.016*np.abs(td);bad=delta>limit
 native_bad=~torch.isclose(x,t,atol=.016,rtol=.016,equal_nan=True).numpy()
 np.savez_compressed(P/(name+'.npz'),actual=xd,target=td,signed_error=signed,absolute_error=delta,tolerance_limit=limit,violations=bad,torch_dtype_violations=native_bad)
 q=[0,.25,.5,.75,.9,.95,.99,.999,1]
 try:torch.testing.assert_close(x,t,atol=.016,rtol=.016,equal_nan=True);close={'passed':True}
 except AssertionError as e:close={'passed':False,'error':str(e)}
 return {'elements':int(delta.size),'float64_evaluated_violations':int(bad.sum()),'torch_dtype_isclose_violations':int(native_bad.sum()),'unequal':int((xd!=td).sum()),'max_abs':float(delta.max()),'mean_abs':float(delta.mean()),'rms':float(np.sqrt((signed*signed).mean())),'abs_quantiles':dict(zip(map(str,q),map(float,np.quantile(delta,q)))),'all_element_distribution':name+'.npz','original_tolerance_assert_close':close}
out={'utc':datetime.now(timezone.utc).isoformat(),'new_canonical_input':True,'historical_input_identity_established':False,'cpu_only':True,'inputs_identical_between_calls':True,'inputs_preserved_bytes':True,'torch_version':torch.__version__,'torch_config':torch.__config__.show(),'original_tolerance':{'atol':.016,'rtol':.016,'direction':'actual candidate/FP64/FP32 vs target captured original reference'},'output_hashes':{k:save(k,v) for k,v in [('fp64_unrounded',f64),('fp64_rounded',z),('fp32_unrounded',f32),('fp32_rounded',f)]},'comparisons':{}}
for name,x,t in [('candidate_vs_reference',c,r),('fp64_vs_reference',z,r),('fp32_vs_reference',f,r),('candidate_vs_fp64',c,z),('candidate_vs_fp32',c,f),('reference_vs_fp64',r,z),('fp32_vs_fp64',f,z)]:out['comparisons'][name]=compare(x,t,name)
out['torch_npu_imported']='torch_npu' in sys.modules
out['loaded_ascend_libraries']=[l for l in Path('/proc/self/maps').read_text().splitlines() if any(v in l for v in ['libascend','libtorch_npu','libhccl'])]
assert not out['torch_npu_imported'] and not out['loaded_ascend_libraries']
(P/'REFERENCE_COMPARISON.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:{f:v[f] for f in ['float64_evaluated_violations','torch_dtype_isclose_violations','unequal','max_abs']} for k,v in out['comparisons'].items()},indent=2))
