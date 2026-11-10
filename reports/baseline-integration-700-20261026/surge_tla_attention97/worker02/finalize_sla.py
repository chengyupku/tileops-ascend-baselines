from pathlib import Path
import json,datetime,hashlib,collections,ast
P=Path(__file__).resolve().parent
rows=json.loads((P/'RESULT.json').read_text());keys=json.loads((P/'SLA_REMAINING.json').read_text());new=[]
for row in rows:
 if row['key'] not in keys:continue
 for a in row.get('attempts',[]):
  if '-sla-continuation-' not in a['receipt']:continue
  out=Path(a['receipt']).parent
  a['native_call_started']=(out/'CALL_STARTED.json').exists();a['native_synchronized']=(out/'CALL_RETURNED.json').exists()
  a['seed_evidence']=str(out/'checker_seed.json') if (out/'checker_seed.json').exists() else None
  a['guard_log']=str(out/'guard.log')
  a['sla_overdue']=True
  if a['guard_exit']==124:
   a['reason']='Original foreground command exceeded unchanged 300s budget; no numerical verdict without complete checker.'
   if a['native_call_started'] and not a['native_synchronized']:a['reason']='Original native call entered; synchronization did not complete within 300s case budget. No numerical verdict; no permitted changed condition for replay.'
  if a['guard_exit']==75:a['reason']='Guard queue or physical idle admission expired before child start; not numerical failure.'
  journal=P/'INPUT_ARTIFACT_RELOCATIONS.jsonl'
  records=[json.loads(l) for l in journal.read_text().splitlines()] if journal.exists() else []
  a['input_preservation']=next((r for r in reversed(records) if out.name in r['remote'] and r['remote'].endswith('/inputs.pt')),None)
  native=out/'launch_01/native.json'
  if native.exists():
   ident=json.loads(native.read_text())['binary'];binary=out/'launch_01'/Path(ident['path']).name
   a['native_binary_hash_matches']=binary.exists() and hashlib.file_digest(binary.open('rb'),'sha256').hexdigest()==ident['sha256']
  new.append(dict(key=row['key'],**a))
 row['runtime_ready']=row['registration_ready']=bool(row.get('numerical_success'))
 row['reason']=new[-1].get('reason','See original complete checker and guard receipt')
 row['accepted_added']=0
 (P/'results'/(row['key'][2]+'.json')).write_text(json.dumps(row,indent=2))
assert {tuple(r['key']) for r in rows}=={tuple(r['key']) for r in json.loads((P/'SCOPE.json').read_text())}
(P/'RESULT.json').write_text(json.dumps(rows,indent=2));(P/'SLA_ATTEMPTS.json').write_text(json.dumps(new,indent=2))
d=json.loads((P/'DELIVERY.json').read_text());d['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();d['status_counts']=dict(collections.Counter(r['status'] for r in rows));d['exact_key_results']=[dict(key=r['key'],status=r['status'],numerical_success=r.get('numerical_success',False),accepted_added=0) for r in rows]
attempts=[a for r in rows for a in r.get('attempts',[])];d['total_guard_submissions']=len(attempts);d['actual_native_attempts']=sum(bool(a.get('actual_native_launch')) for a in attempts);d['guard_exit_counts']=dict(collections.Counter(str(a.get('guard_exit')) for a in attempts));d['numerical_successes']=sum(bool(r.get('numerical_success')) for r in rows)
d['sla_continuation']=dict(original_deadline='2026-10-27T05:31:00+00:00',deadline_reset=False,overdue=True,exact_keys=keys,attempts=new,accepted_added=0)
d['continuation_unattempted']=[]
d['remaining']=['No successful complete checker means no READY publication or numerical failure assertion.','No unchanged native timeout replay is authorized; parent reviews exact-key receipts.','accepted_added=0; registry and memory unchanged.']
d['final_guard_status']='See SLA_FINAL_STATUS.json; latest final three-host observation.'
d['files']+=['SLA_CONTRACT.json','SLA_CPU_DELTA.json','SLA_053100_SNAPSHOT.json','SLA_ATTEMPTS.json','SLA_FINAL_STATUS.json','sla_continue.py','finalize_sla.py']
for name in ['queue_seconds','physical_idle_wait_seconds','guarded_command_seconds']:
 d['timings']['known_'+('lock_queue_seconds' if name=='queue_seconds' else name)]=sum(a.get('timings',{}).get(name) or 0 for a in attempts)
(P/'DELIVERY.json').write_text(json.dumps(d,indent=2))
for f in [P/'sla_continue.py',P/'finalize_sla.py']:ast.parse(f.read_text())
(P/'HEARTBEAT.json').write_text(json.dumps(dict(utc=d['utc'],phase='completed_sla_continuation',key=keys[0],last_output='RESULT.json, SLA_ATTEMPTS.json, DELIVERY.json',blocker=d['remaining'],accepted_added=0),indent=2))
files={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in P.iterdir() if f.is_file() and f.name not in ['FILEHASHES.json','FINAL_HASHES.json','HEARTBEAT.json'] and f.stat().st_size<2000000}
(P/'FILEHASHES.json').write_text(json.dumps(files,indent=2));print(json.dumps(dict(status_counts=d['status_counts'],new_attempts=new),indent=2))
