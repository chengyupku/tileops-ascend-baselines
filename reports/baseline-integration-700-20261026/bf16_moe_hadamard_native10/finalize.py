from pathlib import Path
import json,hashlib,datetime,re
T=Path(__file__).resolve().parent
rows=[]
for op,cid in [('MoEGroupedGemmFwdOp','qwen3-30b-ep8-decode-bfloat16'),('MoEGroupedGemmFwdOp','qwen3-30b-ep8-routed256-bfloat16'),('MoEGroupedGemmFwdOp','ds-v3-decode-down-bfloat16'),('MoEGroupedGemmFwdOp','ds-v3-decode-down-bfloat16-float32'),('MoEGroupedGemmFwdOp','ds-v3-ep8-masked-bfloat16'),('MoEGroupedGemmFwdOp','ds-v3-prefill-down-bfloat16'),('MoEGroupedGemmFwdOp','ds-v3-prefill-mlp-up-bfloat16'),('HadamardTransformFwdOp','llama-8b-prefill-bfloat16'),('HadamardTransformFwdOp','llama3-head-dim-bfloat16')]:
 d=T/cid; rep=d/'representative.json'; guard=d/'guard.log';
 if rep.exists():
  x=json.loads(rep.read_text()); v=x.get('verification',{}); status='verified' if x.get('status')=='verified' else 'failed'
  reason=None
 elif cid=='ds-v3-decode-down-bfloat16': status='blocked';reason='resource envelope unavailable before input allocation; MoE family gate stopped remaining MoE cases'
 else: status='blocked';reason='not run after MoE family gate or dependency scheduling; no numerical claim'
 rows.append({'key':['ops-transformer',op,cid],'status':status,'reason':reason,'registered':False,'accepted_added':0,'verified_unregistered':status=='verified','checker_receipt':str(rep.relative_to(T)) if rep.exists() else None,'guard_receipt':str(guard.relative_to(T)) if guard.exists() else None,'verification':v if rep.exists() else None})
result={'worker':'b264c3aec50f4825','scope':'SCOPE10 MoE8+Hadamard2','accepted_added':0,'verified_unregistered':[r['key'] for r in rows if r['status']=='verified'],'blocked_or_failed':[r for r in rows if r['status']!='verified'],'cpu_checks':json.loads((T/'CPU_CHECKS.json').read_text()) if (T/'CPU_CHECKS.json').exists() else json.loads((T.parent/'bf16_moe_hadamard_parent10'/'CPU_CHECKS.json').read_text()),'deployment':json.loads((T.parent/'DEPLOYMENT_RELEASE_BF16_V5_20261026T2333.json').read_text()),'rows':rows,'generated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(T/'RESULT.json').write_text(json.dumps(result,indent=2,default=str))
(T/'DELIVERY.json').write_text(json.dumps({'worker':result['worker'],'accepted_added':0,'verified_unregistered':result['verified_unregistered'],'blocked_or_failed':result['blocked_or_failed'],'result':'RESULT.json','cpu_checks':'CPU_CHECKS.json','frozen':'FROZEN.json','preparation':'PREPARATION.json','scope_note':'No shared registry or parent state modified.'},indent=2,default=str))
files={str(p.relative_to(T)):hashlib.sha256(p.read_bytes()).hexdigest() for p in T.rglob('*') if p.is_file() and p.name!='FILEHASHES.json'}
(T/'FILEHASHES.json').write_text(json.dumps(files,indent=2))
