import json,datetime,hashlib,ast
from pathlib import Path
P=Path(__file__).resolve().parent
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(name,data):
 f=P/(name+'.tmp');f.write_text(json.dumps(data,indent=2));f.replace(P/name)
d=json.loads((P/'RESULT.json').read_text());g=P/'other01/GUARD_RESULT.json';interference={'detected_at':'2026-10-26T14:27:19.439663+00:00','external_pid':2808453,'external_uid':1003,'external_owner':'unknown/unmanaged (not this guard process group)','guard_returncode':76,'guard_receipt':str(g),'guard_log':str(P/'other01/guard.log'),'status_receipt':str(P/'POST_INTERFERENCE_STATUS.json'),'process_receipt':str(P/'EXTERNAL_PROCESS.txt'),'action':'No further NPU reservation submitted; CPU-only finalization. No kill/reset or identical retry.'}
atomic('INTERFERENCE.json',interference)
for r in d['rows']:
 if r['status']=='verified_unregistered':continue
 if r.get('source_changes'):continue
 if r.get('guard_receipt')==str(g):
  r['intermediate_checker_outcome']=r['reason'];r['reason']='Batch guard76 invalidated acceptance. External UID1003 PID2808453 appeared at 14:27:19.439663 UTC. Intermediate checker results are not counted as admissible numeric verdicts; no identical rerun.'
 else:
  r['reason']='Not executed in this attempt: stopped new local device requests after external UID1003 PID2808453 interrupted other01 with guard76. FP32 route already passed other shapes; this key has no numerical verdict and is not declared unsupported.'
  r['guard_receipt']=str(g);r['evidence']=[str(P/'INTERFERENCE.json'),str(P/'other01/guard.log')];r['attempt']=None
 r['blocked_by']=interference
 d['updated_at']=now
atomic('RESULT.json',d)
atomic('HEARTBEAT.json',dict(utc=now,phase='delivery_cpu_only_after_guard76',current_keys=[],last_progress_at=now,blocked_reason='External UID1003 PID2808453 guard76; provider registration is outside this scope write authorization.',counts={'verified_unregistered':7,'blocked':16,'completed':0}))
# Prove all seven accepted tests ran precisely the same frozen implementation.
expected=hashlib.sha256((P/'adapter.py').read_bytes()).hexdigest();checks=[]
for r in d['rows']:
 if r['status']!='verified_unregistered':continue
 f=Path(r['changed_files'][0]);assert hashlib.sha256(f.read_bytes()).hexdigest()==expected
 c=json.loads(Path(r['checker_receipt']).read_text());assert c['status']=='verified' and c['inputs_preserved'] and not c['oracle_source_changed'] and not c['provider_source_changed'];assert c['verification']['checked_outputs']==c['verification']['total_outputs']==1
 assert json.loads(Path(r['guard_receipt']).read_text())['returncode']==0
 assert str(f.relative_to('/root/dyq/workspace')) in c['source_sha256']
 checks.append(dict(key=[r['baseline'],r['operator'],r['case_id']],frozen_adapter_sha256=expected,checker_verified=True,guard_zero=True,all_outputs=True))
atomic('ACCEPTANCE_AUDIT.json',dict(rows=checks,repeated_numerical_success_runs=0,source_changed_not_accepted=True,guard76_not_accepted=True))
print('final rows',len(d['rows']),'accepted',len(checks))
