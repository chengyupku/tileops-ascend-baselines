from pathlib import Path
import json,datetime,hashlib,ast
W=Path(__file__).resolve().parent
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
new=[]
for a in sorted(W.glob('attempt*')):
 if not a.is_dir() or not (a/'RESULT.json').exists():continue
 r=json.loads((a/'RESULT.json').read_text());c=json.loads((a/'checker.json').read_text()) if (a/'checker.json').exists() else {};g=r['guard_metadata'];t=r['timings']
 if g.get('owner')!='baseline700-20261026/surge_dense_linear92/worker04':
  r['other_owner_lock_snapshot']=g;g={};r['guard_metadata']={};t.update(requested=None,acquired=None)
 def delta(x,y):return (datetime.datetime.fromisoformat(x)-datetime.datetime.fromisoformat(y)).total_seconds() if x and y else None
 t.update(queue_seconds=delta(g.get('acquired_at'),g.get('requested_at')),total_guard_seconds=delta(t['end'],g.get('requested_at')),checker_seconds=c.get('wall_seconds'),physical_wait_seconds=None,child_start_utc=None,compile_seconds=0)
 t['submission_elapsed_seconds']=delta(t['end'],json.loads((a/'STARTED.json').read_text())['utc'])
 if not c:
  r['status']='started_aborted_guard_visibility' if r['guard_exit']==78 and g.get('child') else ('not_run_guard_wait_timeout' if r['guard_exit']==75 else 'not_run_guard_visibility')
  r['reason']=g.get('failure_snapshot',{}).get('kind','Guard did not produce a completed canonical checker receipt')
 r.update(new_device_execution=bool(g.get('child')),task_type='actual supplementary check',checked_outputs=2 if c.get('status')=='verified' else 0,outputs_comparison_attempted=2 if c.get('status')=='verified' else (1 if c.get('stage')=='verify' else 0),total_outputs=2,outputs_captured=0,checker_short_circuit=c.get('stage')=='verify' and c.get('status')=='failed',guard_completed_uncontaminated=g.get('phase')=='finished' and r['guard_exit'] in (0,1) and not g.get('conflict_detected',False),provider_source_changed=c.get('provider_source_changed'),oracle_source_changed=c.get('oracle_source_changed'),input_preservation=c.get('inputs_preserved','Not certified: original verifier short-circuits before completion'),native_identity=str(a/'POST_AUDIT.json'),source_build_mapping='Inherited official package evidence; exact source-to-package build mapping unproven; not READY')
 if (a/'RESOURCE_BOUND.json').exists():
  r['status']='blocked_native_workspace_capacity';r['resource_bound']=str(a/'RESOURCE_BOUND.json');r['outputs_comparison_attempted']=0;r['checker_short_circuit']=False
 if (a/'POST_AUDIT.json').exists():
  p=json.loads((a/'POST_AUDIT.json').read_text());r['frozen_files_unchanged']=p['frozen_files_unchanged'];r['native_binaries']=p['actual_dispatched_binaries'];r['outputs_captured']=2 if any(Path(x['path']).name=='outputs.pt' for x in p.get('captures',[])) else 0
 (a/'RESULT.json').write_text(json.dumps(r,indent=2));new.append(r)
prior=json.loads((W/'PRIOR_REVIEW.json').read_text());by={tuple(r['key']):r for r in prior+new};scope=json.loads((W/'SCOPE.json').read_text());rows=[]
for s in scope:
 r=by.get(tuple(s['key']),dict(key=s['key'],status='not_run',reason='No completed attempt receipt',registered=False,accepted_added=0));r['prior_scope_path']=s['prior_scope_path'];r['attempts']=[{k:x.get(k) for k in ('status','guard_exit','checker','guard','new_device_execution','timings')} for x in new if x['key']==r['key']];rows.append(r)
(W/'RESULT.json').write_text(json.dumps(rows,indent=2))
for p in (W/'package').glob('*.py'):ast.parse(p.read_text())
summary=dict(utc=now(),scope=len(scope),new_attempts=len(new),workloads_started=sum(x['new_device_execution'] for x in new),canonical_completed=sum(x['status'] in ('failed','verified') for x in new),new_verified=sum(x['status']=='verified' for x in new),registered=0,accepted_added=0,prior_failures=sum(x['status']=='prior_exact_checker_failed_no_retry' for x in prior),resource_gated=sum(x['status']=='not_run_resource_gate' for x in prior),checks='Exact keys verified; Python syntax parsed; original source hashes and all-output capture reviewed; no kernel/checker/tolerance/seed edits',remaining='See per-key RESULT; failures require original implementation correction, resource-gated keys require full unchanged checker capacity',timings={'queue_seconds':sum(x['timings']['queue_seconds'] or 0 for x in new),'total_guard_seconds':sum(x['timings']['total_guard_seconds'] or 0 for x in new),'checker_seconds':sum(x['timings']['checker_seconds'] or 0 for x in new),'submission_elapsed_seconds':sum(x['timings']['submission_elapsed_seconds'] or 0 for x in new),'device_only_seconds':None,'physical_wait_seconds':None,'held_cpu_seconds':None})
(W/'SUMMARY.json').write_text(json.dumps(summary,indent=2));(W/'HEARTBEAT.json').write_text(json.dumps(dict(utc=now(),phase='complete',last_output='RESULT.json SUMMARY.json',exact_keys=[x['key'] for x in rows],next='supervisor delivers; parent reviews')))
report=['# Worker04 supplementary baseline checks','',json.dumps(summary,ensure_ascii=False),'','| Baseline | Operator | Case | Result |','|---|---|---|---|']
for x in rows:report.append('| '+' | '.join(x['key']+[x['status']])+' |')
report+=['','Evidence: RESULT.json, each attempt/RESULT.json, checker.json (when completed), raw.log, FROZEN.json, LOCK_FINAL.json and POST_AUDIT.json. Remote actual input/output tensors remain under the same worker directory on server2; captured paths/sizes are in POST_AUDIT.json.','', 'Only package operational owner/entry-root paths were rehomed. Original mathematical binding, binaries, checker, oracle, seed and tolerance were unchanged. Shared npu-guard changed externally during the first attempts; exact hash delta is retained. No registrations or READY claim.','', 'MoE bounds describe conditional full input-preservation failure diagnostics, not a measured universal execution peak. These keys were not numerically tested. Missing server1 recipe dependencies are enumerated in SERVER1_RECIPE_MISSING.json; local x86_64 cannot reuse aarch64 binary.','', 'Guard 75 = no start by deadline. Guard 78 with child = incomplete execution aborted on visibility loss. Original checker failed at attention and short-circuited, so second outputs are captured but not canonically passed. Device-only/held-CPU/physical-wait time are unknown.','', 'Next: resolve host /proc visibility and compatible execution window for unfinished keys; preserve original numerical failures unless a legal original implementation/interface condition changes. Parent owns evidence review and registration.']
(W/'REPORT.md').write_text('\n'.join(report)+'\n')
print(json.dumps(summary,indent=2))
