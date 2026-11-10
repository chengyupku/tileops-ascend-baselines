from pathlib import Path
import json,collections,hashlib,datetime
P=Path(__file__).resolve().parent
scope={tuple(r['key']) for r in json.loads((P/'SCOPE.json').read_text())};rows=[];checks={};bad=[]
def walk(x):
 if isinstance(x,dict):
  p=x.get('path');h=x.get('sha256')
  if isinstance(p,str) and isinstance(h,str) and Path(p).is_file() and p not in checks:
   checks[p]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
   if checks[p]!=h:bad.append(p)
  for v in x.values():walk(v)
 elif isinstance(x,list):
  for v in x:walk(v)
for name in ['original_ops','vllm_tilelang','fla_recurrent','sgl_recurrent_attention','sgl_moe_comm','bmm_policy_review']:
 f=P/name/'RESULT.json';x=json.loads(f.read_text());walk(x)
 for r in x.get('rows',x.get('results',[])):
  rows.append({'key':r['key'],'status':r['status'],'evidence':str(f),'runtime_ready':False,'ready_to_publish':False,'new_numerical_run':False,'required_change':r.get('blocking_change_required',r.get('required_change',x.get('required_changes'))),'source_finding':r.get('finding',r.get('current_source_finding',r.get('reason'))),'private_adapter':str(P/name/r['private_adapter']) if r.get('private_adapter') else None})
keys=[tuple(r['key']) for r in rows];assert len(keys)==71 and len(set(keys))==71 and set(keys)==scope;assert not bad
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
x={'at':now,'scope_count':71,'covered_exactkeys':71,'duplicates':0,'scope_mismatch':False,'accepted_added':0,'new_numerical_passes':0,'npu_execution_count':0,'ready_to_publish':0,'review':'Original-source interfaces, historical exact receipts, current fixed-host deployment and source identities reviewed; not a device validation','source_hashes_checked':len(checks),'source_hash_mismatches':bad,'status_counts':dict(collections.Counter(r['status'] for r in rows)),'rows':rows,'ongoing_cpu_work':'quant_deploy_bundle prepares missing original-source closure; no device authorization','authority_limits':{'writes':'group report directory and PLAN.json only; no remote mutation authorization','npu':'forbidden including initializing imports','publish':'acceptance_root only','original_math_checker_immutable':True}}
(P/'RESULT.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
(P/'REVIEW_SOURCE_CHECKS.json').write_text(json.dumps({'at':now,'files_checked':len(checks),'mismatches':bad,'sha256':checks},ensure_ascii=False,indent=2)+'\n')
(P/'PUBLISH_QUEUE.json').write_text(json.dumps({'at':now,'entries':[],'reason':'No admissible original full-checker success candidate. Historical Bmm3 composition cannot be published. CPU preparation does not meet publication contract.'},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in x.items() if k not in ['rows','status_counts','authority_limits']},ensure_ascii=False))
