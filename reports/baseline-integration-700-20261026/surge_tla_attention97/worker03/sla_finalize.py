from pathlib import Path
import json,datetime,hashlib,collections
from orchestrate import P,OWNER
TAG='sla_resume_20261027T0529';out=P/TAG
# Reconstruct every original and continuation receipt without scheduler side effects.
exec(compile((P/'summarize.py').read_text().split('# Continuation scheduler checkpoint:')[0],str(P/'summarize.py'),'exec'))
r=json.loads((P/'RESULT.json').read_text());new=[]
for row in r['rows']:
 for a in row['attempts']:
  ck=json.loads(Path(a['checker']).read_text()) if a.get('checker') else {}
  a['numerical_failure_established']=ck.get('stage')=='verify' and ck.get('error_type')=='AssertionError'
  if TAG in a['command']:new.append(dict(key=row['key'],**a))
 row['numerical_failure_established']=any(a['numerical_failure_established'] for a in row['attempts'])
 if row['numerical_failure_established']:row['status']='original_numerical_check_failed'
 elif row['attempts'] and row['attempts'][-1]['checker_status']=='device_sync_failed':row['status']='exact_device_sync_failed_no_numerical_verdict'
 row['all_outputs_verified']=row['numerical_success']
r.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sla_deadline='2026-10-27T05:31:00+00:00',sla_overdue=True)
(P/'RESULT.json').write_text(json.dumps(r,indent=2))
d=dict(utc=r['utc'],task_type=r['task_type'],owner=OWNER,scope_count=r['scope_count'],numerical_successes=r['numerical_successes'],accepted_added=0,runtime_ready=False,registered=False,rows=r['rows'],this_native_attempts=new,sla_deadline=r['sla_deadline'],sla_overdue=True,sla_snapshot=str(out/'SLA_0531_SNAPSHOT.json'),remaining_unverified=[x['key'] for x in r['rows'] if not x['numerical_success']],reuse=['DEPLOYMENT_RECIPE.json','CPU_ROUTE_CHECK.json','CONTINUATION_CPU_CHECKS.json'],final_device_status=str(out/'FINAL_DEVICE_STATUS.json'),changed_files=['sla_resume.py','sla_resume_case.py','sla_finalize.py',TAG+'/*','attempts/*-'+TAG+'/*','RESULT.json','DELIVERY.json','FILEHASHES.json','HEARTBEAT.json','REMAINING.json','CHECKS.json'],remaining_issue='Full original checker success required; guard timeout/synchronization failures are not mathematical failure conclusions. No kernel/compiler/checker/seed/tolerance edits.',missing_timing_policy='Unobserved physical idle, held CPU and device execution remain null; controller and host clocks are not mixed.')
(P/'DELIVERY.json').write_text(json.dumps(d,indent=2));(P/'REMAINING.json').write_text(json.dumps(d['remaining_unverified'],indent=2))
(P/'HEARTBEAT.json').write_text(json.dumps(dict(utc=r['utc'],phase='completed_delivery',current_key=None,last_output=str(P/'DELIVERY.json'),blocker=d['remaining_issue'],next='parent original evidence review; accepted_added=0'),indent=2))
files=list(P.glob('*.py'))+[P/n for n in ['RESULT.json','DELIVERY.json','HEARTBEAT.json','CHECKS.json','REMAINING.json']]+list(out.glob('*.json'))
files+=list(P.glob('*'+TAG+'-pre-*.json'))+list(P.glob('sla_resume_final_status-*.json'))
for a in new:
 files+=list(Path(a['command']).parent.glob('*.json'))+[Path(a['guard_log'])]
 if a['checker']:files.append(Path(a['checker']))
 if a['native_evidence']:files.append(Path(a['native_evidence']))
 if a['input_snapshot']:files.append(Path(a['input_snapshot']))
(P/'FILEHASHES.json').write_text(json.dumps(dict(utc=r['utc'],sha256={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in set(files) if f.is_file()},coverage='Updated scripts, full merged delivery, new exact attempt command guard checker native receipts; captured binary identities retained inside per-launch evidence'),indent=2))
print(json.dumps(dict(scope=r['scope_count'],new_attempts=len(new),verified=r['numerical_successes'],accepted_added=0,statuses=dict(collections.Counter(x['status'] for x in r['rows'])))))
