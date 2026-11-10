import os,sys,resource,json,hashlib,time,importlib.abc
from pathlib import Path
P=Path(__file__).resolve().parent
OLD=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_bmm_input_backend1')
C=OLD/'square-4k-bfloat16'
resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
resource.setrlimit(resource.RLIMIT_CPU,(110,115))
assert sys.flags.isolated and sys.flags.no_site
assert os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']=='0'
class BlockDevice(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ['torch_npu','acl','tbe','fla_npu']:raise RuntimeError('Forbidden device import '+fullname)
sys.meta_path.insert(0,BlockDevice())
def audit(event,args):
 if event=='open' and isinstance(args[0],str):
  s=args[0]
  if s.endswith('.pth') or s.startswith(('/dev/davinci','/dev/devmm','/dev/hisi')):raise RuntimeError('Forbidden access '+s)
sys.addaudithook(audit)
sys.path[:0]=['/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages','/opt/tileops-ci/runtime/lib/python3.12/site-packages']
import torch
assert '+cpu' in torch.__version__
def safety():
 maps=Path('/proc/self/maps').read_text();bad=[l for l in maps.splitlines() if any(x in l.lower() for x in ['torch_npu','libascend','libacl_op','libhccl'])];assert not bad,bad
 modules=[m for m in sys.modules if m.startswith(('torch_npu','acl','tbe','fla_npu'))];assert not modules,modules
 fds=[]
 for f in Path('/proc/self/fd').iterdir():
  try:target=os.readlink(f)
  except FileNotFoundError:continue
  if any(x in target for x in ['/dev/davinci','/dev/devmm','/dev/hisi']):fds.append(target)
 assert not fds,fds
 return dict(forbidden_modules=modules,device_fds=fds,forbidden_mappings=bad,mapped_files=sorted(set(l.split()[-1] for l in maps.splitlines() if '/' in l)))
def settings():
 return dict(mkldnn_enabled=torch.backends.mkldnn.enabled,mkldnn_deterministic=torch.backends.mkldnn.deterministic,fp32_precision=torch.backends.mkldnn.matmul.fp32_precision,float32_matmul_precision=torch.get_float32_matmul_precision(),threads=torch.get_num_threads(),torch=torch.__version__,git=torch.version.git_version)

import math
manifest=json.loads((C/'CAPTURE.json').read_text())
torch.set_num_threads(4)
before=safety();original=settings()
hashes={};inputs=[]
for name in ['cpu_reference_input0','cpu_reference_input1']:
 raw=bytearray((C/(name+'.bin')).read_bytes());digest=hashlib.sha256(raw).hexdigest();assert digest==manifest[name]['sha256'];hashes[name]=digest
 t=torch.frombuffer(raw,dtype=torch.bfloat16).reshape(manifest[name]['shape'])
 assert list(t.shape)==[4,4096,4096] and t.is_contiguous()
 inputs.append(t)
a=inputs[0][0,:32,:].double().tolist()
b=inputs[1][0,:,:32].T.double().tolist()
start=time.monotonic()
# BF16 values widen exactly; products of two finite BF16 values are exactly
# representable in FP64 here. math.fsum independently sums each full-K dot.
math64=torch.tensor([[math.fsum(x*y for x,y in zip(row,col)) for col in b] for row in a],dtype=torch.float64)
elapsed=time.monotonic()-start
rounded=math64.to(torch.bfloat16)
outputs={};output_hashes={}
for name,file in [('native','captured_candidate_window'),('default','default'),('disabled','disabled'),('captured_reference','captured_reference_window')]:
 raw=bytearray((OLD/(file+'.bin')).read_bytes());assert len(raw)==64*64*2
 output_hashes[file]=hashlib.sha256(raw).hexdigest()
 outputs[name]=torch.frombuffer(raw,dtype=torch.bfloat16).reshape(64,64)[:32,:32].clone()
assert torch.equal(outputs['default'],outputs['captured_reference'])
def metrics(t):
 x=t.double();d=(x-math64).abs();dr=(x-rounded.double()).abs()
 return dict(elements=x.numel(),max_absolute_error=float(d.max()),mean_absolute_error=float(d.mean()),rms_error=float(d.square().mean().sqrt()),outside_original_tolerance_vs_math64=int((d>(.016+.016*math64.abs())).sum()),unequal_to_rounded_math64=int((t!=rounded).sum()),outside_original_tolerance_vs_rounded_math64=int((dr>(.016+.016*rounded.double().abs())).sum()),max_absolute_error_vs_rounded_math64=float(dr.max()))
errors={k:(v.double()-math64).abs() for k,v in outputs.items()}
pairs={}
for first,second in [('native','default'),('disabled','default'),('native','disabled')]:
 e,f=errors[first],errors[second]
 pairs[first+'_vs_'+second]=dict(first_closer=int((e<f).sum()),second_closer=int((f<e).sum()),equal_distance=int((e==f).sum()))
rows=[]
for i in range(32):
 for j in range(32):rows.append(dict(i=i,j=j,math64=float(math64[i,j]),rounded_bf16=float(rounded[i,j]),**{k:float(v[i,j]) for k,v in outputs.items()}))
(P/'WINDOW_VALUES.json').write_text(json.dumps(rows))
result=dict(status='diagnostic_only',accepted_added=0,window=dict(batch=[0,1],M=[0,32],N=[0,32],K=[0,4096]),method='Independent Python math.fsum of FP64 products of exact BF16 input values; no torch.bmm invoked; captured 64x64 outputs sliced to 32x32',input_hashes=hashes,captured_window_hashes=output_hashes,metrics={k:metrics(v) for k,v in outputs.items()},pairwise_closeness=pairs,math_seconds=elapsed,bf16_rounding=dict(max_abs=float((rounded.double()-math64).abs().max()),mean_abs=float((rounded.double()-math64).abs().mean())),original_tolerance=dict(atol=.016,rtol=.016,comparison='abs(actual-reference) <= atol + rtol*abs(reference)',diagnostic_reference='FP64 or rounded FP64 for analysis only; never substituted into formal oracle'),settings=original,safety_before=before,safety_after=safety(),limits=dict(wall_seconds=120,address_space_bytes=8*1024**3,cpu_soft_seconds=110),no_fullcase_acceptance=True)
(P/'DIAGNOSTIC.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
