from pathlib import Path
import json,datetime,hashlib,ast,collections
P=Path(__file__).resolve().parent
archive=P/'continuation_20261027T0508_prefinal';archive.mkdir(exist_ok=True)
for n in ['RESULT.json','DELIVERY.json','REMAINING.json','CHECKS.json','HEARTBEAT.json']:
 if (P/n).exists() and not (archive/n).exists():(archive/n).write_bytes((P/n).read_bytes())
# Reuse receipt aggregation only; do not invoke the old scheduler side effect.
s=(P/'summarize.py').read_text().split('# Continuation scheduler checkpoint:')[0]
exec(compile(s,str(P/'summarize.py'),'exec'))
r=json.loads((P/'RESULT.json').read_text())
for row in r['rows']:
 for a in row['attempts']:
  ck=json.loads(Path(a['checker']).read_text()) if a.get('checker') else {}
  a['numerical_failure_established']=ck.get('status')=='failed' and ck.get('stage')=='verify' and ck.get('error_type')=='AssertionError'
 row['numerical_failure_established']=any(a['numerical_failure_established'] for a in row['attempts'])
 if not row['numerical_success']:
  if row['numerical_failure_established']:row['status']='original_numerical_check_failed'
  elif row['attempts'] and row['attempts'][-1].get('checker_status')=='device_sync_failed':row['status']='exact_device_sync_failed_no_numerical_verdict'
  elif row['status']=='individually_unverified' or row['status']=='guard_requests_only_no_child_launch':
   row['remaining_reason']='Exact key not executed; prior H64 device synchronization failures are a shared original-route concern, not an exact-key failure. No changed permitted implementation condition; remaining wait600/run600 request cannot fit cutoff.'
  else:row['remaining_reason']='Existing exact execution lacks complete successful original checker; unchanged retry is prohibited.'
 row['all_outputs_verified']=row['numerical_success']
r['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(P/'RESULT.json').write_text(json.dumps(r,indent=2))
rows=[]
for row in r['rows']:
 attempts=[]
 for a in row['attempts']:
  attempts.append({k:a.get(k) for k in ['host','command','guard','guard_exit','checker','checker_status','native_evidence','input_snapshot','actual_child_started','requested_at_controller','requested_at_host','acquired_at_host','child_started_at_host','end_at_controller','timings','changed_condition']})
 rows.append({'key':row['key'],'status':row['status'],'numerical_success':row['numerical_success'],'numerical_failure_established':row['numerical_failure_established'],'reason':row.get('remaining_reason',row.get('reason')),'attempts':attempts,'prior_scope_path':row['prior_scope_path'],'prior_exact_evidence':row.get('prior_exact_evidence')})
new=[a for row in rows for a in row['attempts'] if 'continuation_20261027T0508' in a['command']]
d={'utc':r['utc'],'task_type':'actual_retest_and_interface_integration','owner':r['owner'],'scope_count':len(rows),'accepted_added':0,'numerical_successes':r['numerical_successes'],'registered':False,'runtime_ready':False,'rows':rows,'this_native_new_attempts':new,'inherited_request':'mamba2-2.7b-decode-b256-bfloat16-libpathfix01; old controller and remote guard observed alive on entry; allowed to complete without duplicate','remaining_unverified':[x['key'] for x in rows if not x['numerical_success']],'unstarted_exact_keys':[x['key'] for x in rows if not any(a['actual_child_started'] for a in x['attempts']) and not x.get('prior_exact_evidence')],'checked_sources':'CHECKS.json','deployment_recipe':'DEPLOYMENT_RECIPE.json','cpu_route_checks':'CPU_ROUTE_CHECK.json','modified_implementation':False,'missing_timing_policy':'Unobserved physical_idle_wait, held_cpu, device_execution remain null; controller wall and host lock queue are recorded separately; host clocks differ, no cross-host subtraction','guard_success_is_not_pair_success':True,'remaining_issue':'No successful full original checker: not publishable or accepted. No original kernel/compiler/checker/seed/tolerance changes allowed.'}
d['changed_files_this_native']=['continuation_20261027T0508_case.py','continuation_20261027T0508_launch.py','continuation_20261027T0508_finalize.py','continuation_20261027T0508_*.json','continuation_20261027T05*_status-*.json','attempts/ds-v2-4k-bfloat16-continuation_20261027T0508/','RESULT.json','DELIVERY.json','FILEHASHES.json','HEARTBEAT.json','REMAINING.json','CHECKS.json']
d['final_device_observation']={'snapshot_prefix':'continuation_20261027T0521_finalstatus','all_three_status_exit':0,'old_server2_guard_and_child_absent':True,'new_server1_child_absent':True,'evidence':'continuation_20261027T0521_exit-*.json','no_device_requests_left_for_this_worker':True}
(P/'DELIVERY.json').write_text(json.dumps(d,indent=2))
(P/'REMAINING.json').write_text(json.dumps(d['remaining_unverified'],indent=2))
(P/'HEARTBEAT.json').write_text(json.dumps({'utc':r['utc'],'current_key':None,'phase':'completed_delivery','last_output':str(P/'DELIVERY.json'),'blocker':d['remaining_issue'],'next':'parent review; no shared registry writes','accepted_added':0},indent=2))
files=list(P.glob('*.py'))+[P/n for n in ['SCOPE.json','CONTINUATION_SCOPE.json','RESULT.json','DELIVERY.json','HEARTBEAT.json','CHECKS.json','DEPLOYMENT_RECIPE.json','CPU_ROUTE_CHECK.json']]
for a in [a for row in rows for a in row['attempts']]:
 for k in ['command','guard','checker','native_evidence']:
  if a.get(k):files.append(Path(a[k]))
files.extend((P/'original_sources').rglob('*.py'))
files.extend(P.glob('continuation_20261027T05*.json'))
files.extend((P/'attempts').glob('*/guard.log'))
hashes={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(set(files)) if f.is_file()}
(P/'FILEHASHES.json').write_text(json.dumps({'utc':r['utc'],'coverage':'Private scripts, final delivery and per-attempt command/guard/checker/native receipts; runtime binary identities remain in capture receipts','sha256':hashes},indent=2))
print(json.dumps({'scope':len(rows),'new_attempts':len(new),'verified':r['numerical_successes'],'accepted_added':0,'statuses':dict(collections.Counter(x['status'] for x in rows))}))
