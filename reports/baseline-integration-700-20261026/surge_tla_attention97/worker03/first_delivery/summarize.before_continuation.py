from pathlib import Path
import json,datetime,hashlib,re,ast
P=Path(__file__).resolve().parent
rows=[]
for original in json.loads((P/'SCOPE.json').read_text()):
 key=original['key'];cid=key[2];attempts=[]
 for f in (P/'attempts').glob('*/command.json'):
  c=json.loads(f.read_text())
  if c['key']!=key:continue
  d=f.parent;g=d/'guard.json';g=json.loads(g.read_text()) if g.exists() else None
  raw=(d/'guard.log').read_text(errors='replace') if (d/'guard.log').exists() else ''
  evidence=d/'runs'/(cid+'-pto-A5-original-branch') if c['host']!='local' else P/'local_runs'/(cid+'-pto-A5-original-branch')
  ck=evidence/'checker.json';ckdata=json.loads(ck.read_text()) if ck.exists() else {}
  meta=evidence/'GUARD_RUNNING.json';m=json.loads(meta.read_text()) if meta.exists() else {}
  call=evidence/'CALL_STARTED.json';call=json.loads(call.read_text()) if call.exists() else {}
  a={'host':c['host'],'command':str(f),'guard':str(d/'guard.json') if g else None,'guard_log':str(d/'guard.log'),'guard_exit':g.get('returncode') if g else None,'requested_at_controller':c.get('requested_at'),'requested_at_host':m.get('requested_at'),'acquired_at_host':m.get('acquired_at'),'child_started_at_host':m.get('updated_at'),'end_at_controller':g.get('ended_at') if g else None,'child':m.get('child'),'actual_child_started':bool(m),'native_call':bool(call),'native_evidence':str(evidence/'native.json') if (evidence/'native.json').exists() else None,'checker':str(ck) if ck.exists() else None,'checker_status':ckdata.get('status'),'error':ckdata.get('error'),'input_snapshot':str(evidence/'inputs.pt') if (evidence/'inputs.pt').exists() else None,'changed_condition':c.get('changed_condition'),'timings':{'controller_wall_seconds':g['end']-g['start'] if g else None,'lock_queue_seconds':None,'physical_idle_wait_seconds':None,'held_cpu_seconds':None,'device_execution_seconds':None}}
  if m:
   a['timings']['lock_queue_seconds']=(datetime.datetime.fromisoformat(m['acquired_at'])-datetime.datetime.fromisoformat(m['requested_at'])).total_seconds()
   a['timings']['acquired_to_child_seconds']=(datetime.datetime.fromisoformat(m['updated_at'])-datetime.datetime.fromisoformat(m['acquired_at'])).total_seconds()
  failures=[]
  for line in raw.splitlines():
   if line.startswith('[npu-guard] failure_snapshot '):
    try:failures.append(json.loads(line.split('failure_snapshot ',1)[1]))
    except json.JSONDecodeError:pass
  a['guard_failure_snapshots']=failures
  a['verified']=a['guard_exit']==0 and ckdata.get('status')=='verified'
  attempts.append(a)
 attempts.sort(key=lambda a:a['requested_at_controller'])
 r={'key':key,'prior_scope_path':original['prior_scope_path'],'attempts':attempts,'numerical_success':any(a['verified'] for a in attempts),'accepted_added':0,'registered':False,'runtime_ready':False,'numerical_failure_established':False}
 if r['numerical_success']:r['status']='verified_pending_parent_review'
 elif any(a['actual_child_started'] for a in attempts):r['status']='actual_execution_without_valid_full_checker'
 elif attempts:r['status']='guard_requests_only_no_child_launch'
 elif cid=='mamba2-2.7b-decode-b1-bfloat16':r.update(status='prior_exact_timeout_retained',prior_exact_evidence=str(P/'prior_exact_b1/RESULT.json'),reason='Original exact guard124 retained; unchanged retry not submitted')
 else:r.update(status='individually_unverified',reason='No exact device result this round; no sibling failure substituted')
 rows.append(r)
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'task_type':'actual_retest_and_interface_integration','owner':'baseline700-20261026/surge_tla_attention97/worker03','scope_count':len(rows),'accepted_added':0,'numerical_successes':sum(r['numerical_success'] for r in rows),'rows':rows}
(P/'RESULT.json').write_text(json.dumps(result,indent=2))
(P/'CHECKS.json').write_text(json.dumps({'unique_exact_keys':len({tuple(r['key']) for r in rows}),'script_ast_valid':all(ast.parse(f.read_text()) is not None for f in P.glob('*.py')),'private_source_original_matches':{str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest()==hashlib.sha256((Path('/root/dyq/workspace/ascend_baselines/tilelang-ascend')/f.relative_to(P/'original_sources')).read_bytes()).hexdigest() for f in (P/'original_sources').rglob('*.py')},'kernel_compiler_checker_seed_tolerance_modified':False},indent=2))
print(json.dumps({'keys':len(rows),'attempts':sum(len(r['attempts']) for r in rows),'actual_child_starts':sum(a['actual_child_started'] for r in rows for a in r['attempts']),'verified':result['numerical_successes'],'accepted':0}))
