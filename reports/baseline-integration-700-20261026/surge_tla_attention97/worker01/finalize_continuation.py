from pathlib import Path
import json,datetime,hashlib,collections
P=Path(__file__).resolve().parent;D=P/'continuation_20261027T043900'
s=json.loads((P/'RESULT.json').read_text());now=datetime.datetime.now(datetime.timezone.utc).isoformat()
for f in sorted(D.glob('*.RESULT.json'),key=lambda f:json.loads(f.read_text()).get('utc','')):
 a=json.loads(f.read_text());r=next(x for x in s['exactkeys'] if x['key']==a['key']);g=a['guard'];out=Path(a['evidence']);c=a['checker']
 r.setdefault('continuation_attempts',[])
 entry=dict(host=a['host'],guard=str(out/'guard.json'),checker=str(out/'checker.json') if c else None,guard_exit=g.get('returncode'),elapsed_with_queue=g.get('end',0)-g.get('start',0),admission=str(D/(r['key'][2]+'.'+a['host']+'.admission.json')) if (D/(r['key'][2]+'.'+a['host']+'.admission.json')).exists() else str(D/(r['key'][2]+'.admission.json')))
 if entry not in r['continuation_attempts']:r['continuation_attempts'].append(entry)
 r.update(guard=g,checker=c,host=a['host'],evidence=str(out),numerical_success=False,accepted_added=0,runtime_ready=False)
 r['native_evidence']=[str(f) for f in sorted(out.glob('**/native.json'))]
 r['input_evidence']=str(out/'inputs.pt') if (out/'inputs.pt').exists() else str(out/'REMOTE_INPUT_RECEIPT.json') if (out/'REMOTE_INPUT_RECEIPT.json').exists() else None
 r['input_preservation_verified']=False
 r['registration_preparation']='private adapter and deployment recipe present; not READY without complete original checker pass'
 rc=g.get('returncode')
 if c:r['status']='checker_pass_pending_parent_review' if c.get('status')=='passed' else 'original_checker_error_exact_attempt'
 elif rc==76:r['status']='external_competition_guard76_not_accepted'
 elif rc==124:r['status']='exact_pto_timeout_not_numerical_failure'
 elif rc in [75,78]:r['status']=f'not_started_guard{rc}'
 else:r['status']='exact_attempt_incomplete_or_error'
 if c:r['error_tail']=c.get('error','').splitlines()[-3:]
 (P/(r['key'][2]+'.RESULT.json')).write_text(json.dumps(r,indent=2))
# One physical request per host and guard receipt, independent of admission path spelling.
for r in s['exactkeys']:
 unique={}
 for a in r.get('continuation_attempts',[]):unique[(a['host'],str(Path(a['guard']).resolve()))]=a
 r['continuation_attempts']=list(unique.values())
# Reconcile historical exact attempts without replacing unknowns by representative outcomes.
for r in s['exactkeys']:
 if r['key'][2] in ['ds-v3-single-request-float16','serving-kv512-float16','speculative-decode-float16']:
  r['status']='prior_exact_pto_timeout_retained';r['reason']='Prior exact source mapping and receipt retain actual PTO native timeout; no changed legal execution condition. Historical default-target compile evidence is retained separately.'
 elif isinstance(r.get('checker'),dict) and not r.get('continuation_attempts'):
  r['status']='original_codegen_ascend_fill_exact_attempt' if 'Unresolved call Op(tl.ascend_fill)' in r['checker'].get('error','') else 'original_checker_error_exact_attempt'
  if r.get('guard',{}).get('metadata',{}).get('conflict_detected'):r['status']+='__guard_conflicted'
 r['accepted_added']=0;r['runtime_ready']=False
s.update(utc=now,accepted_added=0,numerical_successes=0,scope_count=len(s['exactkeys']),status_counts=dict(collections.Counter(r['status'] for r in s['exactkeys'])))
(P/'RESULT.json').write_text(json.dumps(s,indent=2))
for r in s['exactkeys']:(P/(r['key'][2]+'.RESULT.json')).write_text(json.dumps(r,indent=2))
t=[]
def epoch(v):
 try:return datetime.datetime.fromisoformat(v).timestamp()
 except:return None
for r in s['exactkeys']:
 for a in r.get('new_attempts',[])+r.get('continuation_attempts',[]):
  f=Path(a['guard']);g=json.loads(f.read_text());m=g.get('metadata',{});run=f.parent;running=json.loads((run/'GUARD_RUNNING.json').read_text()) if (run/'GUARD_RUNNING.json').exists() else {};m={**running,**m};req=epoch(m.get('requested_at'));acq=epoch(m.get('acquired_at'));child=epoch(running.get('updated_at'))
  t.append(dict(key=r['key'],host=a['host'],guard_exit=g.get('returncode'),requested_at=m.get('requested_at'),request_epoch=g.get('start'),acquired_at=m.get('acquired_at'),child_start=running.get('updated_at'),end_epoch=g.get('end'),lock_queue_seconds=acq-req if acq is not None and req is not None else None,physical_wait_seconds=child-acq if child is not None and acq is not None else None,command_and_cleanup_seconds=g['end']-child if child is not None else None,device_seconds=None,held_cpu_seconds=None,total_seconds=g.get('end',0)-g.get('start',0),evidence=str(f)))
(D/'TIMINGS.json').write_text(json.dumps(t,indent=2))
print(json.dumps(s['status_counts']))
