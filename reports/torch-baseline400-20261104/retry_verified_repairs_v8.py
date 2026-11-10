"""One-time exact release after v2 representative batch, only matching diagnosed errors."""
import pathlib,json,time,hashlib
p=pathlib.Path(__file__).resolve().parent;h=p/'server1'
assert (p/'SOURCES.v8.server1.json').exists()
state=json.loads((h/'STATE.json').read_text());retry=json.loads((h/'RETRY.json').read_text()) if (h/'RETRY.json').exists() else {};changes=[]
for tid,s in state.items():
 op=s['key'][1];d=h/'runs'/tid/s['attempt'];log=(d/'device.log').read_text(errors='replace') if (d/'device.log').exists() else ''
 errs=' '.join(f.read_text() for f in d.glob('*failure.json'));reason=None
 if op=='SSDChunkStateFwdOp' and s['status']=='failed':
  reason='v8 evaluates same SSD formula on independent batch/chunk slices; original128GiB outer-product temporary exceeds verified96GiB cgroup limit, prepare exited-9; checker and reduction axis unchanged'
 elif op=='KDAFwdOp' and s['status']=='completed':
  reason='Correct copied KDA constant to frozen original _REF_CHUNK=16; initial chunk32 sample retained as experiment, not campaign baseline'
  (d/'STATE.before-kda-constant-audit.json').write_text(json.dumps(s,indent=2));s['status']='invalid_adapter_constant'
 elif op=='GemmINT8W8A8FwdOp' and 'npu_cpu_fallback' in log:
  reason='v5 exact INT32 contraction using bounded512-term FP32 dot on NPU replaces unsupported _int_mm CPU fallback; preserve independent CPU checker'
  if s['status']=='completed':
   (d/'STATE.before-cpu-fallback-audit.json').write_text(json.dumps(s,indent=2));s['status']='invalid_cpu_fallback'
 elif 'DT_FLOAT8' in errs and ('Index' in errs or 'index' in errs):reason='v4 byte-preserving FP8 indexing/scatter adapter; CPU bytes test passed; all operations timed'
 elif op=='FusedMoESharedExpertFwdOp' and 'capacity' in errs:reason='v3 lossless compressed256MiB storage shards preserve views and52GiB disk reserve; CPU roundtrip validated'
 if reason:
  retry[tid]={'at':time.time(),'reason':reason,'source_sha256':hashlib.sha256((p/'SOURCES.v8.server1.json').read_bytes()).hexdigest()};changes.append({'id':tid,**retry[tid]})
for n,v in [('STATE.json',state),('RETRY.json',retry),('V8_RETRY_ACTIONS.json',changes)]:
 f=h/n;t=f.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(f)
print(json.dumps(changes))
