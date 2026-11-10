from pathlib import Path
import json,hashlib,datetime
P=Path(__file__).resolve().parent;W=Path('/opt/tileops-all-pairs/workspace')
scope=json.loads((P/'SCOPE.json').read_text())
libs=list(Path('/opt/tileops-all-pairs').rglob('libtilelang_module.so'))
record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),tilelang_vendor=str(W/'tilelang'),built_libraries=[str(x) for x in libs],rows=[])
for r in scope:
 if r['baseline']=='tilelang':
  reason='No built upstream tilelang Ascend runtime found on assigned server1; discovered DSOs belong to tilelang-ascend compiler/runtime, not this baseline. Building a full separate compiler is outside the authorized minimal deployment repair.'
  api=r['evidence']
  if r['operator']=='NSAVarlenFwdOp':reason='Original parallel_nsa_fwd unconditionally calls torch.cuda.get_device_capability() and invokes an autotuned CUDA-oriented kernel. No public device-capability argument; this is not an executable unchanged Ascend top-level API.'
  if r['operator']=='SigmoidFwdOp':reason='Proposed generic run_single_arg_mathop_test hardcodes target=cuda and returns no callable; former composed sigmoid requires forbidden multi-kernel math. No complete Ascend sigmoid entry found in targeted examples/ascend or testing/ascend searches.'
  record['rows'].append(dict(baseline=r['baseline'],operator=r['operator'],case_id=r['case_id'],status='blocked',reason=reason,api=api))
for src in ['examples/deepseek_nsa/example_triton_nsa_fwd_varlen.py','examples/deepseek_mhc/example_mhc_post.py','examples/ascend/example_per_token_cast_to_fp8.py','examples/ascend/example_gemm.py','testing/python/math/test_math_fast_math.py']:
 f=W/'tilelang'/src
 if f.exists():
  d=P/'source_evidence'/src;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(f.read_bytes())
(P/'STATIC_BLOCKERS.json').write_text(json.dumps(record,indent=2))
print('TileLang deployment/source blocked keys:',len(record['rows']))
