from pathlib import Path
import json,datetime,collections,fcntl
P=Path(__file__).resolve().parent;B=P.parent
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
accepted={tuple(x['key']):x for x in json.loads((B/'ACCEPTED.json').read_text())['rows']}
rows=[]
for lib,op,cid in json.loads((P/'SCOPE.json').read_text()):
 base='runs_retry' if lib=='vllm-ascend' and cid=='q35-27b-32x8192-float16' else 'runs_fla_binary_retry' if lib=='flash-linear-attention-npu' else 'runs'
 f=P/base/lib/cid/'checker.json';d=json.loads(f.read_text()) if f.exists() else {}
 k=(lib,op,cid)
 status='verified_registered' if k in accepted else 'failed_original_compilation' if lib=='vllm-ascend' and cid=='q35-27b-32x8192-float16' else 'failed_original_numerical' if lib=='vllm-ascend' and d.get('error_type')=='AssertionError' else d.get('status','pending')
 rows.append({'key':list(k),'accepted':k in accepted,'status':status,'checker':str(f),'error':d.get('error'),'reference_device':d.get('reference_device'),'verification':d.get('verification'),'wall_seconds':d.get('wall_seconds'),'source_unchanged':not d.get('provider_source_changed',True) and not d.get('oracle_source_changed',True)})
for cid in ['llama-70b-prefill-float16','llama-8b-prefill-float16']:
 f=P/'runs_addln_original/ops-nn'/cid/'checker.json';d=json.loads(f.read_text())
 rows.append({'key':['ops-nn','FusedAddLayerNormFwdOp',cid],'accepted':False,'status':'original_precision_contract_mismatch','checker':str(f),'error':d['error'],'contract_change_applied':False,'actual_70b_diagnostic':str(P/'ADD_LAYERNORM_ACTUAL_DIAGNOSIS.json'),'source_unchanged':not d['provider_source_changed'] and not d['oracle_source_changed']})
summary={'at':now,'scope':13,'accepted_this_request':sum(x['accepted'] for x in rows),'unaccepted_this_request':sum(not x['accepted'] for x in rows),'server1_total_scope':42,'server1_accepted':len(accepted),'server1_remaining':42-len(accepted),'rows':rows,'formal_checker':str(P/'FORMAL_CHECKER_APPLIED.json'),'tests':str(P/'CPU_DISPATCH_TESTS.json'),'addln_semantic_proposal':str(P/'add_layernorm_reference.PROPOSED.diff'),'contract_decision':'Required before any reference precision change; proposal UNAPPLIED.'}
(P/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
with open(B/'STATE.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 q=B/'RESULT.json';result=json.loads(q.read_text());updates={tuple(x['key']):x for x in rows}
 for row in result['rows']:
  if tuple(row['key']) in updates:
   update=updates[tuple(row['key'])];row.setdefault('before_fp64_addln_fix',dict(row));row.update(update);row['current_review']=str(P/'SUMMARY.json')
 result.update(at=now,accepted_added=len(accepted),remaining_unaccepted=42-len(accepted),status_counts=dict(collections.Counter(x['status'] for x in result['rows'])));q.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='rows'},ensure_ascii=False,indent=2))
