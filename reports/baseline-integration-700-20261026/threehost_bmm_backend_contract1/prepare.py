import subprocess,json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
host='openlibing-DevEnv-661495'
R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/'
old=R+'threehost_bmm_input_backend1'
new=R+P.name
base=subprocess.check_output([ssh,host,'cat '+old+'/cpu_diagnostic.py'],text=True)
(P/'reused_cpu_isolation_source.txt').write_text(base)
prefix=base.split('manifest=json.loads')[0].replace("C=P/'square-4k-bfloat16'", "OLD=Path("+repr(old)+")\nC=OLD/'square-4k-bfloat16'")
body=r'''
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
'''
(P/'cpu_fp64_diagnostic.py').write_text(prefix+body)
static=r'''
import json,hashlib
from pathlib import Path
root=Path('/opt/tileops-all-pairs/workspace')
old=root/'reports/baseline-integration-700-20261026/threehost_bmm_input_backend1'
torch=root/'.cache/all-pairs/remote_triton_runtime/overlay/site-packages/torch'
files=[(old/'TileOPs/workloads/gemm.py',[(364,389)]),(old/'TileOPs/scripts/ascend_pair_check.py',[(137,162),(320,345)]),(old/'run.sh',[(1,100)]),(old/'FROZEN.json',[(1,250)]),(torch/'backends/mkldnn/__init__.py',[(1,160)]),(torch/'include/ATen/Context.h',[(1,9999)]),(torch/'_torch_docs.py',[(1,99999)])]
out={}
for p,ranges in files:
 raw=p.read_bytes();lines=raw.decode().splitlines(); excerpts=[]
 if p.name=='Context.h':
  ranges=[(max(1,i-3),min(len(lines),i+5)) for i,l in enumerate(lines,1) if any(t in l for t in ['enabled_mkldnn','allowBF16Reduction','float32_matmul_precision','Float32Precision::NONE'])]
 if p.name=='_torch_docs.py':
  hits=[i for i,l in enumerate(lines,1) if l.strip()=='torch.bmm,'];ranges=[(i,i+85) for i in hits]
 for lo,hi in ranges:excerpts.append(dict(first=lo,text='\n'.join(lines[lo-1:hi])))
 out[str(p)]=dict(sha256=hashlib.sha256(raw).hexdigest(),excerpts=excerpts)
print(json.dumps(out,indent=2))
'''
(P/'collect_static.py').write_text(static)
r=subprocess.run([ssh,host,'/usr/bin/python3 -I -S -B -'],input=static,text=True,capture_output=True,check=True)
(P/'STATIC_EVIDENCE.json').write_text(r.stdout)
subprocess.run([ssh,host,'mkdir -p '+new],check=True)
subprocess.run([ssh,host,'cat > '+new+'/cpu_fp64_diagnostic.py'],input=prefix+body,text=True,check=True)
print('Prepared diagnostic and static evidence; numerical run not yet started')
