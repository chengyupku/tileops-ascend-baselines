from pathlib import Path
import json,datetime,re,hashlib,subprocess
P=Path(__file__).resolve().parent
scope=json.loads((P/'SCOPE.json').read_text());reg=json.loads((P/'REGISTRATION.json').read_text()) if (P/'REGISTRATION.json').exists() else {'keys':[]};registered={tuple(k) for k in reg['keys']}
rows=[];attempts={}
for d in sorted(P.glob('case[0-9][0-9]')):
 cmd=json.loads((d/'COMMAND.json').read_text());attempts[cmd[-2]]=d
stop_reason=None
for s in scope:
 key=s['case_id'];d=attempts.get(key)
 row={'baseline':'ops-nn','operator':'GemmFwdOp','case_id':key,'numerical_pass':False,'registered':('GemmFwdOp',key) in registered,'parent_accepted_added':0}
 if d is None:
  row.update(status='not_run',reason='Stopped after preceding failure or incomplete evidence; no numerical verdict for this key.')
 else:
  g=json.loads((d/'GUARD_RESULT.json').read_text());c=json.loads((d/'checker.json').read_text()) if (d/'checker.json').exists() else {}
  frozen=d/'FROZEN.json';gm=json.loads((d/'GUARD_METADATA.json').read_text()) if (d/'GUARD_METADATA.json').exists() else {}
  dt=lambda s:datetime.datetime.fromisoformat(s)
  compile_s=[]
  if frozen.exists():
   f=json.loads(frozen.read_text())
   for e in f['native_compiles']:
    m=re.search(r'/0:(\d+):(\d+\.\d+)',e['text'])
    if m:compile_s.append(int(m[1])*60+float(m[2]))
  good=frozen.exists() and g['returncode']==0 and c.get('status')=='verified'
  row.update(status='registered_pending_parent_review' if good and row['registered'] else 'numerically_verified' if good else 'attempt_failed',numerical_pass=good,reason=None if good else c.get('error') or (d/'guard.log').read_text()[-2500:],guard_returncode=g['returncode'],checker=str(d/'checker.json') if c else None,guard=str(d/'GUARD_RESULT.json'),frozen_evidence=str(frozen) if frozen.exists() else None,input_seed=c.get('input_seed'),case_snapshot=c.get('case_snapshot'),verification=c.get('verification'),inputs_preserved=c.get('inputs_preserved'),provider_source_changed=c.get('provider_source_changed'),oracle_source_changed=c.get('oracle_source_changed'),timings={'guard_total_s':g['elapsed_s'],'lock_queue_s':(dt(gm['acquired_at'])-dt(gm['requested_at'])).total_seconds() if gm else None,'idle_check_and_launch_s':(dt(gm['updated_at'])-dt(gm['acquired_at'])).total_seconds() if gm else None,'checker_total_s':c.get('wall_seconds'),'native_compile_s':compile_s,'device_only_s':None},raw_tensor_files=[str(p) for p in d.glob('*.pt')])
 rows.append(row)
start=min(datetime.datetime.fromisoformat(json.loads((d/'GUARD_RESULT.json').read_text())['requested_at']) for d in attempts.values());end=datetime.datetime.now(datetime.timezone.utc)
result={'at':end.isoformat(),'scope_count':8,'attempted':len(attempts),'numerical_pass':sum(r['numerical_pass'] for r in rows),'registered':sum(r['registered'] for r in rows),'parent_accepted_added':0,'parent_baseline_at_assignment':312,'failed':sum(r['status']=='attempt_failed' for r in rows),'not_run':sum(r['status']=='not_run' for r in rows),'rows':rows,'reproduction':{'host':'local aa','owner':'baseline700-20261026/nn_dense_resume8','command_receipts':[str(d/'COMMAND.json') for d in attempts.values()],'vendor':'/root/dyq/workspace/.cache/ascend-classic/native-pilot-20261020/ops-nn/install/vendors/classicpilot_nn','snapshot':str(P/'snapshot/TileOPs'),'snapshot_manifest':str(P/'SNAPSHOT.json'),'adapter_sha256':hashlib.sha256((P/'adapter.py').read_bytes()).hexdigest(),'reference_changed':False,'kernel_changed':False,'compiler_target_changed':False,'observation':'Existing checker observe_call wrapped only to save canonical inputs and complete outputs, with synchronization; original verify/reference untouched. CPU registry test uses stub torch/GraphCall/candidate constructors and proves routing only.'},'elapsed_since_first_request_s':(end-start).total_seconds(),'remaining_issues':['Parent review and acceptance are separate and have not been performed.'],'changed_files':[str(P),' /root/dyq/workspace/TileOPs/benchmarks/ascend_all_pairs/ops_nn/integration700_nn_dense_resume8.py'.strip(),'/root/dyq/workspace/TileOPs/benchmarks/ascend_all_pairs/ops_nn/__init__.py'] if registered else [str(P)]}
for name in ['RESULT.json','DELIVERY.json']:(P/name).write_text(json.dumps(result,indent=2))
(P/'HEARTBEAT.json').write_text(json.dumps({'at':end.isoformat(),'phase':'completed','numerical_pass':result['numerical_pass'],'registered':result['registered'],'parent_accepted_added':0},indent=2))
print({k:result[k] for k in ['attempted','numerical_pass','registered','failed','not_run','parent_accepted_added','elapsed_since_first_request_s']})
