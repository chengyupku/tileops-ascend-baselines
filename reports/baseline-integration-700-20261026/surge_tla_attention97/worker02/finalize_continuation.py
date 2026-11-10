from pathlib import Path
import json,datetime,hashlib,collections,ast
P=Path(__file__).resolve().parent;rows=json.loads((P/'RESULT.json').read_text());scope=json.loads((P/'SCOPE.json').read_text());checks=[]
for row in rows:
 for a in row.get('attempts',[]):
  out=Path(a['receipt']).parent
  if a.get('guard_exit')==78:
   a['status']='device_visibility_lost_invalid' if 'device visibility lost' in (out/'guard.log').read_text() else 'guard_unknown_or_permission_failure'
  if a.get('actual_native_launch'):
   f=out/'launch_01/native.json'
   if f.exists():
    ident=json.loads(f.read_text())['binary'];binary=out/'launch_01'/Path(ident['path']).name
    checks.append(dict(key=row['key'],binary=str(binary),present=binary.exists(),hash_matches=hashlib.file_digest(binary.open('rb'),'sha256').hexdigest()==ident['sha256'] if binary.exists() else None))
   a['native_call_started']=(out/'CALL_STARTED.json').exists();a['native_synchronized']=(out/'CALL_RETURNED.json').exists()
   a['input_archive']=str(out/'inputs.pt') if (out/'inputs.pt').exists() else None
   a['seed_evidence']=str(out/'checker_seed.json') if (out/'checker_seed.json').exists() else None
 if row.get('attempts'):
  row['status']=row['attempts'][-1]['status']
 if row['key'][2]=='gemma-2b-8k-bs1-float16':row['historical_exact_receipt']=str(P/'HISTORICAL_GEMMA_EXACT.json')
 if row['status']=='pending_actual_check':row['status']='not_run_by_deadline';row['reason']='No successful device reservation/run before bounded batch deadline; not a failure.'
 row['registration_ready']=row.get('numerical_success',False)
 (P/'results').mkdir(exist_ok=True);(P/'results'/(row['key'][2]+'.json')).write_text(json.dumps(row,indent=2))
assert {tuple(x['key']) for x in rows}=={tuple(x['key']) for x in scope}
(P/'RESULT.json').write_text(json.dumps(rows,indent=2))
for f in P.glob('*.py'):ast.parse(f.read_text())
original=P.parents[1]/'threehost_attention_mla_mha31'
frozen=['scripts/ascend_pair_check.py','workloads/numerics.py','benchmarks/api.py','workloads/workload_base.py','src/tileops/manifest/spec/attention.yaml']
frozen_checks={rel:hashlib.sha256((P/'frozen/TileOPs'/rel).read_bytes()).hexdigest()==hashlib.sha256((original/'frozen/TileOPs'/rel).read_bytes()).hexdigest() for rel in frozen}
files={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in P.glob('*') if f.is_file() and f.name not in ['FINAL_HASHES.json','HEARTBEAT.json'] and f.stat().st_size<2000000}
(P/'FINAL_HASHES.json').write_text(json.dumps(files,indent=2))
(P/'CHECKS.json').write_text(json.dumps(dict(exact_scope_matches=True,private_python_ast=True,original_checker_reference_manifest_unchanged=frozen_checks,native_artifact_checks=checks),indent=2))
attempts=[a for r in rows for a in r.get('attempts',[])]
sumtime=lambda k:sum(a.get('timings',{}).get(k) or 0 for a in attempts)
d=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),task_type='actual_integration_and_supplemental_verification',owner='baseline700-20261026/surge_tla_attention97/worker02',scope_count=len(rows),status_counts=dict(collections.Counter(r['status'] for r in rows)),actual_native_attempts=sum(bool(a.get('actual_native_launch')) for a in attempts),total_guard_submissions=len(attempts),guard_exit_counts=dict(collections.Counter(str(a.get('guard_exit')) for a in attempts)),numerical_successes=sum(r.get('numerical_success',False) for r in rows),accepted_added=0,registered=0,timings=dict(known_lock_queue_seconds=sumtime('queue_seconds'),known_physical_idle_wait_seconds=sumtime('physical_idle_wait_seconds'),known_guarded_command_seconds=sumtime('guarded_command_seconds'),unstarted_queue_and_physical_wait_seconds=sumtime('queue_and_physical_wait_seconds'),held_cpu_seconds=None,device_execution_seconds=None,model_queue_seconds=None),files=['RESULT.json','results/','CHECKS.json','CPU_ROUTE.json','CONTRACT.json','RECIPE.md','INPUT_ARTIFACT_RELOCATIONS.jsonl','QWEN_GUARD78_CAUSE.json','HISTORICAL_GEMMA_EXACT.json','FINAL_HASHES.json','status_samples/','local_runs/','openlibing-DevEnv-661495/','openlibing-DevEnv-055409/'],remaining=['Parent reviews original receipts and performs any formal registration; no READY claim without successful complete checker.','Native synchronization stalls have no established permitted source/runtime fix; no unchanged timeout replay.','Guard78 is infrastructure invalidity, not mathematical failure.'],constraints=dict(kernel_compiler_pass_schedule_reference_checker_seed_tolerance_modified=False,global_registry_memory_modified=False,subagents_spawned=False,external_process_killed=False,writes='Only worker02 private directory and matching remote directories.'))
(P/'DELIVERY.json').write_text(json.dumps(d,indent=2));(P/'HEARTBEAT.json').write_text(json.dumps(dict(utc=d['utc'],phase='completed_delivery',key=None,last_output='DELIVERY.json, RESULT.json, CHECKS.json',remaining=d['remaining']),indent=2));print(json.dumps({k:d[k] for k in ['scope_count','status_counts','total_guard_submissions','numerical_successes','accepted_added']}))

# Continuation-specific exact-key closure; never infer numeric failures from missing checker.
continuation=[]
for row in rows:
 for a in row.get('attempts',[]):
  if '-continuation-' not in a['receipt']:continue
  out=Path(a['receipt']).parent
  if a['guard_exit']==124 and a.get('native_call_started') and not a.get('native_synchronized'):
   a['reason']='Original native call entered and returned asynchronously; synchronization did not finish within unchanged case budget. No numerical verdict.'
  if a['guard_exit']==76:a['reason']='External Ascend process appeared; original strict guard interrupted own workload. No numerical verdict.'
  if a['guard_exit']==75:a['reason']='No device command started before queue/idle wait expired.'
  if (out/'inputs.pt').exists():
   a['input_preservation']={'location':str(out/'inputs.pt'),'bytes':(out/'inputs.pt').stat().st_size,'sha256':hashlib.file_digest((out/'inputs.pt').open('rb'),'sha256').hexdigest()}
  else:
   journal=P/'INPUT_ARTIFACT_RELOCATIONS.jsonl'
   receipts=[json.loads(l) for l in journal.read_text().splitlines()] if journal.exists() else []
   a['input_preservation']=next((r for r in reversed(receipts) if out.name in r['remote'] and r['remote'].endswith('/inputs.pt')),None)
  continuation.append(dict(key=row['key'],**a))
 if row['status']=='not_run_by_deadline':row['reason']='No new guard submission could fit authorized wait 600s + original device budget 300s + collection before deadline; not an execution or numerical failure.'
 row['runtime_ready']=bool(row.get('numerical_success'))
 row['accepted_added']=0
 (P/'results'/(row['key'][2]+'.json')).write_text(json.dumps(row,indent=2))
(P/'RESULT.json').write_text(json.dumps(rows,indent=2))
(P/'CONTINUATION_ATTEMPTS.json').write_text(json.dumps(continuation,indent=2))
d['continuation_attempts']=len(continuation)
d['continuation_successes']=sum(a['status']=='verified_pending_parent_registration' for a in continuation)
d['continuation_guard_exits']=dict(collections.Counter(str(a['guard_exit']) for a in continuation))
d['files']+=['CONTINUATION_ATTEMPTS.json','CONTINUATION_CPU_CHECKS.json','CONTINUATION_CONTRACT.json','RECIPE_CONTINUATION.md','FILEHASHES.json']
(P/'DELIVERY.json').write_text(json.dumps(d,indent=2))
files={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in P.glob('*') if f.is_file() and f.name not in ['FILEHASHES.json','FINAL_HASHES.json','HEARTBEAT.json'] and f.stat().st_size<2000000}
(P/'FILEHASHES.json').write_text(json.dumps(files,indent=2))
print(json.dumps(dict(continuation_attempts=len(continuation),guard_exits=d['continuation_guard_exits'])))
