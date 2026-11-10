import os, sys, ast, json, hashlib, math, platform, time
from pathlib import Path
from datetime import datetime, timezone
from functools import wraps
from types import SimpleNamespace
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
sys.dont_write_bytecode=True
class BlockNpu:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch_npu','acl'): raise RuntimeError('CPU_ONLY import denied: '+fullname)
sys.meta_path.insert(0,BlockNpu())
def audit(event,args):
 if event=='ctypes.dlopen' and any(x in str(args).lower() for x in ('ascend','torch_npu','libacl','libhccl')):
  raise RuntimeError('CPU_ONLY dlopen denied')
sys.addaudithook(audit)
import torch
W=Path('/root/dyq/workspace'); P=Path(__file__).resolve().parent
sys.path.insert(0,str(W/'TileOPs'))
from workloads.device import set_run_device,run_device
from workloads.numerics import verify,Request
set_run_device('cpu'); torch.set_num_threads(4)
assert torch.ones(1).device.type=='cpu'
def dump(n,x): (P/n).write_text(json.dumps(x,indent=2,default=str)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def extract(path,names):
 tree=ast.parse(path.read_text()); nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
 assert len(nodes)==len(names)
 ns={'torch':torch,'wraps':wraps}
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
 return ns
G=W/'reports/baseline-integration-700-20261026/gmm_capture49/attempt01/remote_evidence'
meta=json.loads((G/'CAPTURE.json').read_text()); cmp=json.loads((G/'REFERENCE_COMPARISON.json').read_text())
checks={}
def load(n,dtype=torch.bfloat16):
 p=G/(n+'.bin'); m=meta.get(n,cmp['output_hashes'].get(n)); h=sha(p); checks[str(p)]={'expected':m['sha256'],'actual':h,'match':h==m['sha256']}; assert h==m['sha256']
 return torch.frombuffer(bytearray(p.read_bytes()),dtype=dtype).reshape(m['shape'])
a,b=load('candidate_0_input0'),load('candidate_0_input1'); old=load('reference_0_output'); candidate=load('candidate_0_output'); exact=load('fp64_unrounded',torch.float64); rounded=load('fp64_rounded')
oldfn=extract(W/'TileOPs/workloads/gemm.py',['matmul_fp32_reduction'])['matmul_fp32_reduction']
newfn=extract(P/'gemm_cpu_bf16.py',['matmul_fp32_reduction'])['matmul_fp32_reduction']
flags=torch.backends.cuda.matmul
saved=(flags.allow_fp16_reduced_precision_reduction,flags.allow_bf16_reduced_precision_reduction)
new=newfn(a,b); oldlocal=oldfn(a,b)
assert saved==(flags.allow_fp16_reduced_precision_reduction,flags.allow_bf16_reduced_precision_reduction)
assert new.shape==old.shape and new.dtype==old.dtype
with torch.autocast('cpu',dtype=torch.bfloat16): auto=newfn(a,b)
assert torch.equal(auto,new)
def compare(x,y):
 d=(x.double()-y.double()).abs(); bad=d>(.016+.016*y.double().abs())
 try: torch.testing.assert_close(x,y,atol=.016,rtol=.016,equal_nan=True); passed=True
 except AssertionError: passed=False
 return {'float64_threshold_violations':int(bad.sum()),'max_abs':float(d.max()),'unequal':int((x!=y).sum()),'original_tolerance_assert_close':passed}
fsum=[]
for i,j in [(0,0),(75,611),(446,624),(272,201),(1023,1023),(128,512),(511,513),(991,27)]:
 v=math.fsum(float(x)*float(y) for x,y in zip(a[i],b[:,j])); fsum.append({'i':i,'j':j,'fp64_error':abs(v-float(exact[i,j])),'patched_error':abs(v-float(new[i,j]))})
raw=new.contiguous().view(torch.uint8).numpy().tobytes()
(P/'gemm_patched_output.bin').write_bytes(raw)
dump('PATCHED_OUTPUT.json',{'sha256':hashlib.sha256(raw).hexdigest(),'shape':list(new.shape),'dtype':str(new.dtype),'bytes':len(raw)})
# CPU fallback paths and transposed/batched operands: bounded functional probes, not acceptance cases.
probes=[]
for dtype in (torch.float16,torch.float32,torch.float64):
 x=torch.arange(12,dtype=dtype).reshape(3,4)/8; y=x.T
 assert torch.equal(newfn(x,y),oldfn(x,y)); probes.append(str(dtype))
for x,y in [(a.T,b.T),(a[:8,:16].expand(2,-1,-1),b[:16,:7].expand(2,-1,-1))]:
 got=newfn(x,y); torch.testing.assert_close(got,(x.double()@y.double()).bfloat16(),atol=.016,rtol=.016)
dump('GEMM_CPU_VALIDATION.json',{'utc':datetime.now(timezone.utc).isoformat(),'input_provenance':'actual captured canonical NPU inputs; no RNG regeneration','shape':list(new.shape),'dtype':str(new.dtype),'comparisons':{'fp64_vs_captured_reference':compare(rounded,old),'captured_reference_vs_fp64':compare(old,rounded),'patched_vs_fp64':compare(new,rounded),'captured_candidate_vs_patched':compare(candidate,new),'local_original_vs_captured_original':compare(oldlocal,old)},'independent_fsum':fsum,'cuda_flags_unchanged':True,'cpu_autocast_guard':True,'other_cpu_dtypes_unchanged':probes,'transposed_and_batched_smokes':True,'cuda_runtime_tested':False})
# Extract original mathematical functions verbatim, with unchanged numerics module.
gdn_path=W/'TileOPs/workloads/linear_attention/gdn.py'
gdn=extract(gdn_path,['gdn_ref','gated_verification'])
dispatch=extract(P/'gdn_cpu_dispatch.py',['verify_gdn_prefill_on_cpu','observe_call','tensor_metadata'])['verify_gdn_prefill_on_cpu']
# No payload capture exists in supplied GDN delivery tree. Explicit synthetic scheduling test only.
generator=torch.Generator(device='cpu').manual_seed(9817)
q=torch.randn(1,65,2,8,generator=generator,dtype=torch.bfloat16)*.1
k=torch.randn(q.shape,generator=generator,dtype=q.dtype)*.1
v=torch.randn(q.shape,generator=generator,dtype=q.dtype)*.1
g=-torch.rand(1,65,2,generator=generator,dtype=q.dtype); beta=torch.rand(g.shape,generator=generator,dtype=q.dtype)*.5
initial=torch.randn(1,2,8,8,generator=generator)*.01
inputs=(q,k,v,g,beta,initial,None,None,None,None)
expected=gdn['gdn_ref'](*inputs); case=SimpleNamespace(inputs=inputs,reference=gdn['gdn_ref'],verification=gdn['gated_verification'](inputs))
records=[]; calls=[]
def replay(*args):
 assert all(x is y for x,y in zip(args,inputs)); calls.append('original_arguments_received'); return tuple(x.clone() for x in expected)
reset_calls=[]
res=dispatch(torch,case,replay,inputs,lambda:reset_calls.append(True),'synthetic_replay',records)
assert res['synthetic_replay'].checked_outputs==res['synthetic_replay'].total_outputs==2
rejections={}
for target in (0,1):
 def bad(*args):
  out=[x.clone() for x in expected];out[target].zero_();return tuple(out)
 try: dispatch(torch,case,bad,inputs,None,'corrupt',[])
 except AssertionError as e: rejections[str(target)]=str(e)
 else: raise AssertionError('corrupted complete output accepted')
assert run_device()=='cpu'
# Verify restored run device even on reference failure; no allocations to alternate device.
def fail(*args):
 assert run_device()=='cpu'; raise ValueError('intentional test')
case_fail=SimpleNamespace(inputs=inputs,reference=fail,verification=case.verification)
set_run_device('meta')
try: dispatch(torch,case_fail,replay,inputs,None,'failure',[])
except ValueError: pass
assert run_device()=='meta'; set_run_device('cpu')
dump('GDN_CPU_VALIDATION.json',{'utc':datetime.now(timezone.utc).isoformat(),'actual_gdn_capture_available_in_supplied_tree':False,'test_provenance':'synthetic scheduling/validator smoke, not canonical NPU input and not candidate acceptance','reference_math':'unaltered AST-extracted gdn_ref','validator':'unaltered AST-extracted gated_verification plus original numerics module','checked_outputs':2,'output_shapes':[list(x.shape) for x in expected],'output_dtypes':[str(x.dtype) for x in expected],'candidate_original_args_preserved':bool(calls),'reset_invoked':bool(reset_calls),'zeroed_output_rejected':rejections,'run_device_restored_on_failure':True,'observations':records,'npu_candidate_execution_tested':False,'large_q35_not_run':True})
assert not any(n=='torch_npu' or n.startswith('torch_npu.') or n=='acl' for n in sys.modules)
loaded=[l for l in Path('/proc/self/maps').read_text().splitlines() if any(v in l.lower() for v in ('libascend','libtorch_npu','libhccl','libacl.so'))]
assert not loaded
sources=json.loads((P/'SOURCE_HASHES.json').read_text()); unchanged={f:sha(W/f)==h for f,h in sources.items()}; assert all(unchanged.values())
dump('CPU_ENVIRONMENT.json',{'python':sys.executable,'torch':torch.__version__,'architecture':platform.machine(),'config':torch.__config__.show(),'autoload':os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'],'run_device':str(run_device()),'loaded_ascend_libraries':loaded,'shared_source_hashes_unchanged':unchanged})
dump('INPUT_HASH_CHECKS.json',checks)
dump('HEARTBEAT.json',{'utc':datetime.now(timezone.utc).isoformat(),'stage':'CPU validations completed; writing review','npu_used':False})
print(json.dumps(json.loads((P/'GEMM_CPU_VALIDATION.json').read_text())['comparisons'],indent=2))
print('GDN synthetic dispatch: both outputs checked, both corruption controls rejected. Actual GDN capture missing.')
