from pathlib import Path
import json,datetime,hashlib,collections
P=Path(__file__).resolve().parent
C=P/'continuation_20261027T0440'
def read(p):return json.loads(p.read_text())
def dt(s):return datetime.datetime.fromisoformat(s).timestamp() if s else None
scope=read(P/'SCOPE.json')
rows={tuple(r['key']):r for r in read(P/'checkpoint_20261027T043837'/'RESULT.json')}
attempts=[]
for root in [P/'attempts',C/'attempts']:
 for f in sorted(root.glob('*/guard.json'),key=lambda f:read(f)['start']):
  g=read(f);out=f.parent;checks=list(out.rglob('checker.json'));metas=list(out.rglob('GUARD_RUNNING.json'));c=read(checks[0]) if checks else {};m=read(metas[0]) if metas else {}
  rc=g['returncode'];success=rc==0 and c.get('status')=='verified'
  status='verified_pending_parent_review' if success else {75:'guard_not_started',78:'guard_visibility_failure',124:'device_timeout_no_checker',139:'native_process_SIGSEGV_no_checker',143:'interrupted_no_checker'}.get(rc,'checker_'+c.get('status','absent'))
  times=dict(requested_at=m.get('requested_at',g.get('requested_at')),acquired_at=m.get('acquired_at'),child_start=m.get('updated_at'),end=g.get('ended_at'),queue_seconds=dt(m['acquired_at'])-dt(m['requested_at']) if m.get('acquired_at') else None,physical_idle_wait_seconds=None,prelaunch_guard_seconds=dt(m['updated_at'])-dt(m['acquired_at']) if m.get('acquired_at') else None,cpu_compile_seconds=None,device_active_seconds=None,orchestrator_wall_seconds=g['end']-g['start'])
  a=dict(key=g['key'],host=g['host'],argv=g['argv'],guard=str(f),guard_exit=rc,checker=str(checks[0]) if checks else None,status=status,child_started=bool(m),timings=times,changed_condition=g.get('changed_condition'))
  attempts.append(a)
  if root==C/'attempts' or ('original-direct' in out.name):
   r=dict(a,numerical_success=success,numerical_failure_established=c.get('stage')=='verification' and c.get('status')=='failed',runtime_ready=False,registered=False,accepted_added=0,native_evidence=[str(x) for x in out.rglob('native.json')],checker_status=c.get('status'),checker_error=c.get('error'),seed_evidence=str(P/'EXPECTED_SEEDS.json'),environment_evidence=[str(x) for x in out.rglob('runtime_identity.json')],reason=('Original execution exceeded unchanged 300-second device budget; complete checker absent' if rc==124 else status))
   old=rows[tuple(g['key'])]
   r['prior_scope_path']=old.get('prior_scope_path');r['prior_evidence']=old.get('prior_evidence');r['previous_result']=old
   rows[tuple(g['key'])]=r
   if root==C/'attempts':(out/'RESULT.json').write_text(json.dumps(r,indent=2))
ordered=[rows[tuple(s['key'])] for s in scope]
for r in ordered:r['attempts']=[a for a in attempts if a['key']==r['key']]
(P/'RESULT.json').write_text(json.dumps(ordered,indent=2))
(C/'RESULT.json').write_text(json.dumps(ordered,indent=2))
assert len(ordered)==14 and len({tuple(r['key']) for r in ordered})==14
report=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),task_type='actual integration and supplemental execution',owner='baseline700-20261026/surge_tla_attention97/worker04',scope_count=14,accepted_added=0,verified_pending_parent=sum(r.get('numerical_success',False) for r in ordered),registered=0,status_counts=dict(collections.Counter(r['status'] for r in ordered)),continuation_attempts=[a for a in attempts if '/continuation_20261027T0440/' in a['guard']],all_attempts=attempts,real_remaining=[r['key'] for r in ordered if not r.get('numerical_success')],checks=read(C/'CPU_CHECKS.json'),limitations=['No numerical verdict without complete original checker','Original timeout/API/compiler limits preserved without unchanged replay','Queue/physical idle/CPU/device timings not directly observed remain null','Remote and local clocks differ; do not subtract cross-host timestamps','No global registry or memory modifications; parent sole acceptance authority'],evidence_directory=str(C),deployment_recipe=str(P/'DEPLOYMENT.md'),continuation_recipe=str(C/'RECIPE.md'))
for f in [P/'DELIVERY.json',C/'DELIVERY.json']:f.write_text(json.dumps(report,indent=2))
(P/'HEARTBEAT.json').write_text(json.dumps(dict(utc=report['utc'],current_key=None,phase='finished_delivery',last_output=str(P/'DELIVERY.json'),blocker='See exact per-key RESULT',next='Parent review; no unchanged retry',accepted_added=0),indent=2))
files=[f for f in P.iterdir() if f.is_file() and f.name!='FILEHASHES.json']+list(C.glob('*.json'))
(P/'FILEHASHES.json').write_text(json.dumps({str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
print(json.dumps({k:report[k] for k in ['utc','status_counts','accepted_added','verified_pending_parent']}))
