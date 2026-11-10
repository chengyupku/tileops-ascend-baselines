from pathlib import Path
import json,datetime,os
def dt(s): return datetime.datetime.fromisoformat(s).timestamp() if s else None
P=Path(__file__).resolve().parent
scope=json.loads((P/'SCOPE.json').read_text()); rows={tuple(r['key']):r for r in json.loads((P/'PRESERVED_RESULTS.json').read_text())}
for path in sorted((P/'attempts').glob('*/guard.json'),key=lambda f:json.loads(f.read_text())['start']):
 g=json.loads(path.read_text());out=path.parent; key=g['key']; checks=list(out.rglob('checker.json'));metas=list(out.rglob('GUARD_RUNNING.json'))
 c=json.loads(checks[0].read_text()) if checks else {};m=json.loads(metas[0].read_text()) if metas else {}
 success=g['returncode']==0 and c.get('status')=='verified'
 status='verified_pending_parent_review' if success else 'native_process_SIGSEGV_no_checker' if g['returncode']==139 else 'guard_visibility_failure' if g['returncode']==78 else 'guard_not_started' if g['returncode']==75 else 'device_timeout_no_checker' if g['returncode']==124 else 'checker_'+c.get('status','absent')
 r=dict(key=key,status=status,host=g['host'],guard=str(path),checker=str(checks[0]) if checks else None,checker_status=c.get('status'),checker_error=c.get('error'),numerical_success=success,numerical_failure_established=c.get('stage')=='verification' and c.get('status')=='failed',accepted_added=0,runtime_ready=False,registered=False,guard_exit=g['returncode'],native_evidence=[str(x) for x in out.rglob('native.json')],timings=dict(requested_at=m.get('requested_at'),acquired_at=m.get('acquired_at'),child_start=m.get('updated_at'),end=g.get('ended_at'),queue_seconds=dt(m['acquired_at'])-dt(m['requested_at']) if m.get('acquired_at') and m.get('requested_at') else None,physical_idle_wait_seconds=None,prelaunch_guard_seconds=dt(m['updated_at'])-dt(m['acquired_at']) if m.get('updated_at') and m.get('acquired_at') else None,cpu_compile_seconds=None,device_active_seconds=None,orchestrator_wall_seconds=g['end']-g['start']))
 (out/'RESULT.json').write_text(json.dumps(r,indent=2))
 old=rows.get(tuple(key),{});history=old.get('attempts',[])+[dict(host=r['host'],guard=r['guard'],guard_exit=r['guard_exit'],checker=r['checker'],status=r['status'])]
 if r['status']=='guard_not_started' and old.get('native_evidence'):old['attempts']=history
 else:r['attempts']=history;rows[tuple(key)]=r
for s in scope:
 if tuple(s['key']) not in rows:rows[tuple(s['key'])]=dict(key=s['key'],status='pending_exact_run',numerical_success=False,numerical_failure_established=False,accepted_added=0)
ordered=[rows[tuple(s['key'])] for s in scope]
for r,s in zip(ordered,scope):
 r['prior_scope_path']=s.get('prior_scope_path')
 r['prior_evidence']=dict(prior_attempt=s.get('prior_attempt'),prior_exact_evidence=s.get('prior_exact_evidence'),checker=s.get('checker_receipt'),guard=s.get('guard_receipt'))
tmp=P/('RESULT.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(ordered,indent=2));tmp.replace(P/'RESULT.json')
print(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),counts={x:sum(r['status']==x for r in ordered) for x in sorted({r['status'] for r in ordered})})),flush=True)
