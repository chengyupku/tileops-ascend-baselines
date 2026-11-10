from pathlib import Path
import json,hashlib
W=Path(__file__).resolve().parent;R=W.parents[1]
scope=json.loads((W/'SCOPE.json').read_text());rows=[]
for s in scope:
 k=s['key'];case=k[2]
 if s.get('prior_latest_status')=='not_run_family_blocked':continue
 if case=='decode-b1-float16':p=R/'gdn_fp16_capture14/attempt01_decode/checker.json'
 elif k[1]=='FusedAddLayerNormFwdOp':p=Path(s['prior_checker'])
 elif k[1]=='DeltaNetFwdOp':p=R/'sgl_gdn_remote38/runs/probe10--DeltaNetFwdOp--prefill-l2norm-ragged-bfloat16/checker.json'
 else:
  p=R/'moe_fullchecker_scratch5/PER_KEY_BOUNDS.json';d=json.loads(p.read_text());b=next(x for x in d['keys'] if x['case_id']==case)
  row=dict(key=k,status='not_run_resource_gate',reason=s['reason'],bound_bytes=s['bound'],bound_GiB=s['bound_GiB'],bound_details=b,evidence=str(p),evidence_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),registered=False,accepted_added=0,new_device_execution=False,condition_changed=False)
  rows.append(row);continue
 d=json.loads(p.read_text());assert (d['library'],d['op'],d['case_id'])==tuple(k)
 row=dict(key=k,status='prior_exact_checker_failed_no_retry',reason=d.get('error'),checker=str(p),checker_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),checker_status=d['status'],input_seed=d.get('input_seed'),registered=False,accepted_added=0,new_device_execution=False,condition_changed=False)
 rows.append(row)
(W/'PRIOR_REVIEW.json').write_text(json.dumps(rows,indent=2));print([(x['key'],x['status']) for x in rows])
