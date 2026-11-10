from pathlib import Path
import json,datetime,hashlib,collections,shutil
D=Path(__file__).resolve().parent;P=D.parent;out=D/'runs/serving-b2-float16-pto-A5'
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
g=json.loads((out/'guard.json').read_text());c=json.loads((out/'checker.json').read_text()) if (out/'checker.json').exists() else None
running=json.loads((out/'GUARD_RUNNING.json').read_text()) if (out/'GUARD_RUNNING.json').exists() else {};m={**running,**g.get('metadata',{})}
def epoch(x):return datetime.datetime.fromisoformat(x).timestamp() if x else None
req=m.get('requested_at');acq=m.get('acquired_at');child=running.get('updated_at')
t=dict(requested=req,acquired=acq,child_start=child,end_epoch=g['end'],guard_exit=g['returncode'],lock_queue_seconds=epoch(acq)-epoch(req) if acq and req else None,physical_wait_seconds=epoch(child)-epoch(acq) if child and acq else None,held_cpu_seconds=None,device_execution_seconds=None,child_and_cleanup_seconds=g['end']-epoch(child) if child else None,total_seconds=g['end']-g['start'],clock_note='Remote wall clock approximately 93 seconds behind local; durations use only same-host clock; local transport timestamps preserved.')
(D/'TIMINGS.json').write_text(json.dumps(t,indent=2))
backup=D/'premerge';backup.mkdir(exist_ok=True)
for n in ['RESULT.json','DELIVERY.json','DELIVERY.md','FILEHASHES.json','serving-b2-float16.RESULT.json']:shutil.copy2(P/n,backup/n)
s=json.loads((P/'RESULT.json').read_text());r=next(r for r in s['exactkeys'] if r['key']==['tilelang-ascend','MHADecodePagedWithKVCacheFwdOp','serving-b2-float16'])
status='original_checker_error_exact_attempt' if c else 'exact_attempt_incomplete_or_error'
if c and c.get('status')=='passed':status='checker_pass_pending_parent_review'
if not c and g['returncode']==124:status='exact_pto_timeout_not_numerical_failure'
if not c and g['returncode'] in [75,78]:status='not_started_guard'+str(g['returncode']) if not running else 'guard_error_after_child_start'
r.update(status=status,utc=now,guard=g,checker=c,host='openlibing-DevEnv-661495',evidence=str(out),timings=t,numerical_success=bool(c and c.get('status')=='passed' and g['returncode']==0),numerical_failure_established=False,accepted_added=0,runtime_ready=False,input_evidence=str(out/'inputs.pt') if (out/'inputs.pt').exists() else None,input_preservation_verified=False,native_evidence=[str(f) for f in out.glob('**/native.json')],reason='Exact legal remaining case attempted after remote reservation release; original checker, adapter and source unchanged. No unconditional replay.')
r.setdefault('sla_continuation_attempts',[]).append(dict(host=r['host'],guard=str(out/'guard.json'),checker=str(out/'checker.json') if c else None,guard_exit=g['returncode'],admission=str(D/'admission.json'),timings=str(D/'TIMINGS.json')))
if c:r['error_tail']=c.get('error','').splitlines()[-8:]
(P/'serving-b2-float16.RESULT.json').write_text(json.dumps(r,indent=2));(D/'RESULT.json').write_text(json.dumps(r,indent=2))
s.update(utc=now,accepted_added=0,numerical_successes=sum(bool(x.get('numerical_success')) for x in s['exactkeys']),status_counts=dict(collections.Counter(x['status'] for x in s['exactkeys'])))
(P/'RESULT.json').write_text(json.dumps(s,indent=2))
delivery=json.loads((P/'DELIVERY.json').read_text());delivery['sla_continuation']=dict(utc=now,key=r['key'],status=status,accepted_added=0,numerical_success=r['numerical_success'],evidence=str(out),timings=t,remaining_issue='Original checker did not establish an accepted pass' if not r['numerical_success'] else 'Parent review and registration pending',deadline='2026-10-27T05:31:00+00:00',sla_overdue=now>='2026-10-27T05:31:00',changed_files=['RESULT.json','serving-b2-float16.RESULT.json','DELIVERY.json','DELIVERY.md','FILEHASHES.json','HEARTBEAT.json',D.name+'/'])
(P/'DELIVERY.json').write_text(json.dumps(delivery,indent=2));(D/'DELIVERY.json').write_text(json.dumps(delivery['sla_continuation'],indent=2))
with (P/'DELIVERY.md').open('a') as f:f.write('\n\nSLA continuation '+now+'\nExact key: '+repr(r['key'])+'\nResult: '+status+'; accepted_added=0. Evidence: '+str(out)+'. Guard exit: '+str(g['returncode'])+'. Timing: '+str(D/'TIMINGS.json')+'. No kernel/compiler/checker/seed/tolerance change.\n')
(P/'HEARTBEAT.json').write_text(json.dumps(dict(utc=now,phase='complete',key=r['key'],last_substantial_output=str(D/'RESULT.json'),blocker=status if not r['numerical_success'] else None,next='parent aggregate review; no repeat without new legal condition',accepted_added=0,sla_overdue=now>='2026-10-27T05:31:00'),indent=2))
h={}
for f in D.rglob('*'):
 if f.is_file() and 'premerge' not in f.parts:h[str(f.relative_to(P))]=hashlib.sha256(f.read_bytes()).hexdigest()
for n in ['RESULT.json','serving-b2-float16.RESULT.json','DELIVERY.json','DELIVERY.md','HEARTBEAT.json']:h[n]=hashlib.sha256((P/n).read_bytes()).hexdigest()
old=json.loads((P/'FILEHASHES.json').read_text());(D/'FILEHASHES.json').write_text(json.dumps(h,indent=2))
if isinstance(old,dict):old['sla_continuation_files']=h
else:old={'prior_filehashes':old,'sla_continuation_files':h}
(P/'FILEHASHES.json').write_text(json.dumps(old,indent=2));print(json.dumps(dict(status=status,checker=c,timings=t),indent=2))
