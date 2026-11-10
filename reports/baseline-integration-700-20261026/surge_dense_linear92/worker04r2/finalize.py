from pathlib import Path
import json,datetime,hashlib,subprocess,shlex,sys
W=Path(__file__).resolve().parent
sys.path.insert(0,str(W))
import batch
R=batch.R
p=subprocess.run([batch.SSH,batch.HOST,'/opt/tileops-ci/runtime/bin/python '+R+'/post_audit.py'],text=True,capture_output=True,check=True)
(W/'post_audit_final.log').write_text(p.stdout+p.stderr)
code="from pathlib import Path;import hashlib,json;w=Path("+repr(R)+");print(json.dumps([dict(path=str(p),size_bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()) for p in w.glob('attempt*/*.pt')],indent=2))"
p=subprocess.run([batch.SSH,batch.HOST,'/opt/tileops-ci/runtime/bin/python -c '+shlex.quote(code)],text=True,capture_output=True,check=True)
(W/'REMOTE_CAPTURE_HASHES.json').write_text(p.stdout)
rows=[]
for attempt in ['attempt02','attempt03']:
 batch.collect(attempt)
 A=W/attempt;c=json.loads((A/'checker.json').read_text());g=json.loads((A/'GUARD_RESULT.json').read_text());m=json.loads((A/'LOCK_FINAL.json').read_text());s=json.loads((A/'STARTED.json').read_text())
 assert m['owner']=='baseline700-20261026/surge_dense_linear92/worker04r2'
 dt=datetime.datetime.fromisoformat
 row=dict(key=['ops-transformer','GDNFwdOp',c['case_id']],task_type='补测/实际对接',status='canonical_checker_failed' if c['status']=='failed' else 'verified_pending_parent_review',reason=c.get('error'),new_device_execution=True,checker_receipt=str(A/'checker.json'),guard_receipt=str(A/'GUARD_RESULT.json'),guard_metadata=str(A/'LOCK_FINAL.json'),guard_exit=g['returncode'],native_receipt=str(A/'dispatch.jsonl'),native_log=str(A/'raw.log'),post_use_audit=str(A/'POST_AUDIT.json'),input_seed=c['input_seed'],command=s['command'],remote_captures=batch.HOST+':'+R+'/'+attempt,complete_original_checker_executed=True,all_candidate_outputs_captured=(A/'runtime.maps').exists(),all_outputs_passed=c['status']=='passed',registered=False,accepted_added=0,runtime_ready=False,registration_ready=False,original_source_binary_mapping='Inherited pinned official package; exact source-to-binary build mapping unproven; no READY claim',timings=dict(requested=m.get('requested_at'),acquired=m.get('acquired_at'),child_start=None,end=g['utc'],queue_seconds=(dt(m['acquired_at'])-dt(m['requested_at'])).total_seconds(),held_total_seconds=(dt(m['updated_at'])-dt(m['acquired_at'])).total_seconds(),original_checker_wall_seconds=c['wall_seconds'],physical_wait_seconds=None,device_seconds=None,held_cpu_seconds=None))
 (A/'RESULT.json').write_text(json.dumps(row,indent=2));rows.append(row)
rows.append(json.loads((W/'RESULT_q35_397b.json').read_text()))
(W/'RESULT.json').write_text(json.dumps(rows,indent=2))
(W/'RESULT.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
(W/'DELIVERY.json').write_text(json.dumps(dict(utc=now,scope_complete=True,accepted_added=0,verified_count=sum(r.get('all_outputs_passed',False) for r in rows),new_device_executions=2,exact_key_count=3,results='RESULT.json',registration_ready=False,remaining_unexecuted=[rows[-1]['key']],unexecuted_reason='RESOURCE_BOUND.json: full original checker live bound exceeds physical capacity',checks=['three-host status and CPU recipe compatibility','Python ABI and original API exports','original pinned libraries and oracle unchanged post-use','full original checker executed independently for b128 and b256','actual native dispatch and synchronized return','remote all-input/output captures hashed'],issues=['b128/b256 canonical numerical mismatches; no retry absent changed condition','large key exceeds single-card physical capacity under unchanged checker','exact original source-to-binary mapping inherited but not proven; no READY claim'],changed_files_scope=str(W)),indent=2))
(W/'HEARTBEAT.json').write_text(json.dumps(dict(utc=now,ownership='exclusive scope finished; no remaining processes',stage='complete',last_substantive_output='RESULT.json, DELIVERY.json, FILEHASHES.json',accepted_added=0),indent=2))
files={str(p.relative_to(W)):dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()) for p in W.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='FILEHASHES.json'}
(W/'FILEHASHES.json').write_text(json.dumps(files,indent=2))
print(json.dumps([(r['key'],r['status']) for r in rows]))
