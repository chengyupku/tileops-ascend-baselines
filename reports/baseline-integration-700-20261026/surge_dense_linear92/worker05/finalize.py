from pathlib import Path
import json,datetime,hashlib,sys
W=Path(__file__).resolve().parent;scope=json.loads((W/'SCOPE.json').read_text());now=datetime.datetime.now(datetime.timezone.utc).isoformat();attempts=[]
for p in sorted(W.glob('attempt*/RESULT.json')):
 d=json.loads(p.read_text());a=p.parent
 d.update(host='server2/openlibing-DevEnv-055409',input_preservation_check=None,ready=False)
 if d['guard_exit'] in (75,78) and not (a/'checker.json').exists():
  d.update(status='not_run_guard_queue_timeout' if d['guard_exit']==75 else 'not_run_guard_visibility_refusal',reason='Guard did not launch workload: '+('120 second reservation timeout' if d['guard_exit']==75 else 'incomplete /proc FD visibility, PID 892989'),complete_checker_outputs_saved=False,new_device_execution=False)
 else:
  c=json.loads((a/'checker.json').read_text());d.update(canonical_checker_complete=False,actual_outputs_captured=2,reference_device=c.get('reference_device'),provider_source_changed=c.get('provider_source_changed'),oracle_source_changed=c.get('oracle_source_changed'),negative_controls_completed=False)
  records=[]
  for l in (a/'guard_metadata.jsonl').read_text().splitlines():
   try:
    m=json.loads(json.loads(l)['raw'])
    if m.get('owner')=='baseline700-20261026/surge_dense_linear92/worker05' and m.get('child'):records.append(m)
   except Exception:pass
  if records:d['native_process_identity']=records[0]['child'];d['timings']['child_recorded_at']=records[0].get('updated_at')
 if (a/'POST_AUDIT.json').exists():d['post_audit']=str(a/'POST_AUDIT.json')
 p.write_text(json.dumps(d,indent=2));attempts.append(d)
rows=json.loads((W/'LOGSUMEXP_RESULT.json').read_text())+[json.loads((W/'REUSED_GDN_RESULT.json').read_text())]
for key in [x['key'] for x in scope if x['operator']=='GDNFwdOp' and x['case_id']!='varlen-ragged-grouped-float16']:
 aa=[a for a in attempts if a['key']==key];d=aa[-1].copy();d['attempts']=[dict(checker_receipt=a['checker_receipt'],guard_receipt=a['guard_receipt'],guard_exit=a['guard_exit'],status=a['status'],new_device_execution=a['new_device_execution'],timings=a['timings']) for a in aa];d['prior_scope_path']=next(x['prior_scope_path'] for x in scope if x['key']==key);rows.append(d)
assert len(rows)==13 and len({tuple(x['key']) for x in rows})==13
assert {tuple(x['key']) for x in rows}=={tuple(x['key']) for x in scope}
(W/'RESULT.json').write_text(json.dumps(rows,indent=2));(W/'RESULT.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));(W/'ATTEMPTS.json').write_text(json.dumps(attempts,indent=2))
counts={s:sum(x['status']==s for x in rows) for s in sorted({x['status'] for x in rows})};summary=dict(utc=now,task_type='actual original API integration and missing exact-case tests',keys=13,counts=counts,device_attempts=sum(x['new_device_execution'] for x in attempts),guard_submissions=len(attempts),verified=0,registered=0,accepted_added=0,guard_total_elapsed_s=sum(x['timings'].get('guard_elapsed_seconds') or 0 for x in attempts),measured_queue_s=sum(x['timings'].get('queue_seconds') or 0 for x in attempts),device_seconds=None,held_cpu_seconds=None,physical_wait_seconds=None,timing_note='Queue sum includes only directly recorded acquired-requested intervals; guard75 timeout remains in per-attempt elapsed. CPU reference and device execution cannot be separated.',remaining_keys=[x['key'] for x in rows],unexecuted_gdn_keys=[x['key'] for x in rows if x['operator']=='GDNFwdOp' and x['status'].startswith('not_run')],finished='--complete' in sys.argv)
(W/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
report=['# worker05 actual integration / missing checks','',f"13 exact keys; {summary['device_attempts']} new actual device checks; {len(attempts)} guard submissions. 0 verified, 0 registered, 0 accepted.",'','All keys belong to ops-transformer; no same-name baseline substitution.','', '| Operator | Exact case | Result |','|---|---|---|']
for x in rows:report.append(f"| {x['operator']} | {x['case_id']} | {x['status']} |")
report+=['','Results include per-key commands, seed, environment/identity receipts, actual output capture paths and every guard attempt. Full original checker is unchanged. Numerical failures stop at the first failing attention comparison; both actual outputs were saved, but final-state and negative-control completion are not claimed.','', 'LogSumExp source evidence fixes each exact reduction width against original sp<=16; the disallowed multi-call mathematical reduction tree was not used. The prior varlen-ragged-grouped failure was reused without rerun.','', 'Server2 original ARM recipe was reused. Local x86_64 cannot reuse that ABI; server1 lacks the pinned original deployment paths. Three-host snapshots precede submissions. B512 infrastructure retries have separate changed-condition evidence; no numerical failure was rerun.','', 'Post-use audit found original library/checker/binding pins stable. The shared /usr/local/bin/npu-guard was independently updated during this task; earlier frozen guard file hashes consequently differ from post-use disk state. Both identities and per-attempt metadata are retained; no attempt is claimed accepted. Exact original source-to-binary build mapping remains inherited/unproven.','', f"Guard elapsed sum: {summary['guard_total_elapsed_s']:.3f}s; observed acquired-minus-requested sum: {summary['measured_queue_s']:.3f}s. Device, physical idle wait and held CPU time remain null. Clocks are not mixed.",'', 'Changed files: private package/recipe/runner, transport and result scripts, CONTRACT, deployment instructions, heartbeat, per-attempt receipts/captures, RESULT/SUMMARY/ATTEMPTS and source evidence. No shared registry, original kernel/compiler/reference/checker/seed/tolerance or memory changes.','', 'Next: parent reviews exact receipts. No successful registration module is supplied because no full canonical case passed.']
(W/'REPORT.md').write_text('\n'.join(report)+'\n')
(W/'HEARTBEAT.json').write_text(json.dumps(dict(utc=now,phase='complete' if summary['finished'] else 'partial results collected; final capacity wait active',exact_keys=[x['key'] for x in scope],last_output='RESULT.json; SUMMARY.json; REPORT.md',blocker='No original canonical numerical pass',next='parent review' if summary['finished'] else 'b512 capacity monitor'),indent=2))
print(json.dumps(summary,indent=2))
