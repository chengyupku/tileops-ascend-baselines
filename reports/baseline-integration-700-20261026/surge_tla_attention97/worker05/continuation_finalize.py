from pathlib import Path
import json,datetime,hashlib,collections
P=Path(__file__).resolve().parent
Q=P/'continuation_20261027T0436'
rows=json.loads((P/'RESULT.json').read_text())
new=[]
for r in rows:
 for a in r['attempts']:
  if 'continuation_20261027T0436' in a['guard']:
   new.append(dict(key=r['key'],**a))
counts=dict(collections.Counter(r['status'] for r in rows))
d=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),task_type='实际对接/补测',owner='baseline700-20261026/surge_tla_attention97/worker05',scope_count=len(rows),accepted_added=0,verified_pending_review=sum(r['numerical_success'] for r in rows),registered=0,runtime_ready=False,status_counts=counts,new_attempts=new,all_key_results=str(P/'RESULT.json'),unchanged_prior_receipts='checkpoint_20261027T043546 and original per-run receipts retained',changed_condition='Guard wait 120 -> 600 for unstarted exact keys; no changed kernel/compiler/pass/checker/reference/seed/tolerance',checks=['CPU original source hashes unchanged','Both remote runners/adapters/capture/scope match local SHA256','Three-host guard status before every submit','Old owner absent from all three process samples before continuation','Continuation Python AST syntax valid'],timings=dict(new_guard_wall_seconds=sum(a['timings']['wall_seconds'] for a in new),new_lock_queue_seconds=sum(a['timings']['lock_queue_seconds'] or 0 for a in new),physical_idle_wait_seconds=None,held_cpu_seconds=None,device_execution_seconds=None,model_queue_seconds=None),remaining=[dict(key=r['key'],status=r['status'],reason=r['remaining_reason']) for r in rows if not r['numerical_success']],publication='No READY package without successful complete original checker and guard. Existing deployment recipe remains reproducible but unverified.')
(P/'DELIVERY.json').write_text(json.dumps(d,indent=2))
lines=['# worker05 continuation delivery','',f"Exact scope: {len(rows)}; verified pending review: {d['verified_pending_review']}; accepted_added=0.",'','| Baseline | Operator | Case | Status |','|---|---|---|---|']
for r in rows:lines.append('| '+' | '.join(r['key']+[r['status']])+' |')
lines.extend(['','See RESULT.json for every original/new guard and checker receipt, seeds, native identity and same-host timing. Unknown physical/CPU/device splits remain null.','No kernel/compiler/pass/reference/checker/seed/tolerance changes. Failed and timeout keys remain unregistered.'])
(P/'DELIVERY.md').write_text('\n'.join(lines)+'\n')
files=[P/'RESULT.json',P/'DELIVERY.json',P/'DELIVERY.md',P/'continuation_launcher.py',P/'continuation_orchestrate.py',P/'continuation_finalize.py']
files+=list(Q.glob('*'))
for a in new:
 for name in ['guard','checker','native','guard_log']:
  if a.get(name):files.append(Path(a[name]))
hashes={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.is_file()}
(P/'FILEHASHES.json').write_text(json.dumps(hashes,indent=2))
print(json.dumps(dict(scope=len(rows),statuses=counts,new_attempts=len(new),accepted_added=0)))
