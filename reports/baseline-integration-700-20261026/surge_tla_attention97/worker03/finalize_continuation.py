from pathlib import Path
import json,datetime,subprocess,sys
P=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(P/'summarize.py')],check=True)
r=json.loads((P/'RESULT.json').read_text());remaining=json.loads((P/'REMAINING.json').read_text()); rows=[]
for row in r['rows']:
 a=row.get('attempts',[]);new=[x for x in a if '-continuation' in x['command'] or '-libpathfix' in x['command']]
 summaries=[]
 for x in new:
  checker=json.loads(Path(x['checker']).read_text()) if x.get('checker') else {}
  summaries.append({'host':x['host'],'guard_exit':x['guard_exit'],'child_started':x['actual_child_started'],'native_launch_identity_present':x.get('native_launch_identity_present'),'checker_status':checker.get('status'),'checker_stage':checker.get('stage'),'error_head':str(checker.get('error','')).splitlines()[0][:700] if checker.get('error') else None,'guard':x['guard'],'checker':x['checker'],'command':x['command'],'timings':x['timings'],'requested_at_controller':x['requested_at_controller'],'requested_at_host':x['requested_at_host'],'acquired_at_host':x['acquired_at_host'],'child_started_at_host':x['child_started_at_host'],'end_at_controller':x['end_at_controller']})
 rows.append({'key':row['key'],'status':row['status'],'numerical_success':row['numerical_success'],'registered':False,'continuation_attempts':summaries,'all_history':str(P/'RESULT.json'),'prior_scope_path':row['prior_scope_path']})
unstarted=[x['key'] for x in rows if x['key'] in remaining and not any(a['child_started'] for a in x['continuation_attempts'])]
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'task_type':'actual_retest_and_interface_integration','owner':r['owner'],'scope_count':14,'continuation_scope_count':10,'accepted_added':0,'verified':r['numerical_successes'],'runtime_ready':False,'rows':rows,'remaining_without_continuation_child_start':unstarted,'historical_delivery':str(P/'first_delivery'),'repair':str(P/'LIBPATH_REPAIR.json'),'deployment':str(P/'DEPLOYMENT_RECIPE.json'),'changed_files':['orchestrate.py','adapter.py','summarize.py','continue_remaining.py','finalize_continuation.py','CONTRACT.json','DEPLOYMENT_RECIPE.json','CONTINUATION_PLAN.json','LIBPATH_REPAIR.json','CONTINUATION_CPU_CHECKS.json','CONTINUE_PREFLIGHT_*.json','RESULT.json','CHECKS.json','HEARTBEAT.json','attempts/*-continuation*/','attempts/*-libpathfix*/'],'remaining_issue':'Original full checker has not verified these keys; no registration is authorized by this result. Unstarted keys remain unknown.'}
(P/'CONTINUATION_DELIVERY.json').write_text(json.dumps(d,indent=2));(P/'DELIVERY.json').write_text(json.dumps(d,indent=2))
print(json.dumps({'keys':14,'verified':d['verified'],'new_attempts':sum(len(x['continuation_attempts']) for x in rows),'new_child_starts':sum(a['child_started'] for x in rows for a in x['continuation_attempts']),'unstarted':unstarted}))
