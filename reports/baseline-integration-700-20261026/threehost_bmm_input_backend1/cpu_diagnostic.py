import os,sys,resource,json,hashlib,time,importlib.abc
from pathlib import Path
P=Path(__file__).resolve().parent
C=P/'square-4k-bfloat16'
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
manifest=json.loads((C/'CAPTURE.json').read_text())
inputs=[];hashes={}
for name in ['cpu_reference_input0','cpu_reference_input1']:
 raw=bytearray((C/(name+'.bin')).read_bytes());digest=hashlib.sha256(raw).hexdigest();assert digest==manifest[name]['sha256'];hashes[name]=digest
 inputs.append(torch.frombuffer(raw,dtype=torch.bfloat16).reshape(manifest[name]['shape']))
a=inputs[0][:1,:64,:].contiguous();b=inputs[1][:1,:,:64].contiguous()
torch.set_num_threads(4)
original=settings();assert original['mkldnn_enabled'] is True
before=safety()
print('DIAGNOSTIC_DEFAULT_BEGIN',flush=True);start=time.monotonic();default=torch.bmm(a,b);default_s=time.monotonic()-start;print('DIAGNOSTIC_DEFAULT_END',flush=True)
try:
 torch.backends.mkldnn.enabled=False
 disabled_settings=settings();print('DIAGNOSTIC_DISABLED_BEGIN',flush=True);start=time.monotonic();disabled=torch.bmm(a,b);disabled_s=time.monotonic()-start;print('DIAGNOSTIC_DISABLED_END',flush=True)
finally:torch.backends.mkldnn.enabled=original['mkldnn_enabled']
def compare(a,b):
 d=(a.float()-b.float()).abs();tol=0.016+0.016*b.float().abs()
 return dict(elements=a.numel(),unequal=int((a!=b).sum()),outside_tolerance=int((d>tol).sum()),max_abs=float(d.max()))
def saved_window(name):
 raw=bytearray((C/(name+'.bin')).read_bytes());assert hashlib.sha256(raw).hexdigest()==manifest[name]['sha256']
 return torch.frombuffer(raw,dtype=torch.bfloat16).reshape(4,4096,4096)[:1,:64,:64].clone()
ref=saved_window('reference_0_output');candidate=saved_window('candidate_0_output')
for name,t in [('default',default),('disabled',disabled),('captured_reference_window',ref),('captured_candidate_window',candidate)]:
 (P/(name+'.bin')).write_bytes(t.contiguous().view(torch.uint8).numpy().tobytes())
result=dict(status='diagnostic_only',accepted_added=0,input_hashes=hashes,window={'batch':[0,1],'M':[0,64],'N':[0,64],'K':[0,4096]},original_settings=original,disabled_settings=disabled_settings,restored_settings=settings(),seconds={'default':default_s,'disabled':disabled_s},comparisons={'default_vs_disabled':compare(default,disabled),'default_vs_captured_reference':compare(default,ref),'disabled_vs_captured_reference':compare(disabled,ref),'default_vs_candidate':compare(default,candidate),'disabled_vs_candidate':compare(disabled,candidate)},safety_before=before,safety_after=safety(),limits={'address_space_bytes':8*1024**3,'cpu_soft_seconds':110,'wall_seconds':120},note='Fixed small output window with complete K. No full-case acceptance; window backend may differ from full-shape dispatch. No FP32 matmul oracle.')
(P/'DIAGNOSTIC.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
