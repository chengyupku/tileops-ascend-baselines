from pathlib import Path
import json,datetime,hashlib,collections
P=Path(__file__).resolve().parent
Q=P/'sla_resume_20261027T0529'
rows=json.loads((P/'RESULT.json').read_text())
new=[]
for r in rows:
 for a in r['attempts']:
  if 'sla_resume_20261027T0529' in a['guard']:
   new.append(dict(key=r['key'],**a))
counts=dict(collections.Counter(r['status'] for r in rows))
d=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),task_type='实际对接/补测',owner='baseline700-20261026/surge_tla_attention97/worker05',scope_count=len(rows),accepted_added=0,verified_pending_review=sum(r['numerical_success'] for r in rows),registered=0,runtime_ready=False,status_counts=counts,new_attempts=new,all_key_results=str(P/'RESULT.json'),unchanged_prior_receipts='checkpoint_20261027T043546 and original per-run receipts retained',changed_condition='Only SLA_REMAINING exact keys previously unstarted; fresh three-host admission with 600s wait; original 05:31 SLA retained; no changed kernel/compiler/pass/checker/reference/seed/tolerance',checks=['CPU original source hashes unchanged','Reused prior host equality receipts; actual pre-checker runtime identities collected per attempt','Three-host guard status before every submit','REPLACEMENT_CLEARANCE ownership released; no same-owner active reservation in pre-submit threehost samples','Continuation Python AST syntax valid'],timings=dict(new_guard_wall_seconds=sum(a['timings']['wall_seconds'] for a in new),new_lock_queue_seconds=sum(a['timings']['lock_queue_seconds'] or 0 for a in new),physical_idle_wait_seconds=None,held_cpu_seconds=None,device_execution_seconds=None,model_queue_seconds=None),remaining=[dict(key=r['key'],status=r['status'],reason=r['remaining_reason']) for r in rows if not r['numerical_success']],publication='No READY package without successful complete original checker and guard. Existing deployment recipe remains reproducible but unverified.')
(P/'DELIVERY.json').write_text(json.dumps(d,indent=2))
lines=['# worker05 continuation delivery','',f"Exact scope: {len(rows)}; verified pending review: {d['verified_pending_review']}; accepted_added=0.",'','| Baseline | Operator | Case | Status |','|---|---|---|---|']
for r in rows:lines.append('| '+' | '.join(r['key']+[r['status']])+' |')
lines.extend(['','See RESULT.json for every original/new guard and checker receipt, seeds, native identity and same-host timing. Unknown physical/CPU/device splits remain null.','No kernel/compiler/pass/reference/checker/seed/tolerance changes. Failed and timeout keys remain unregistered.'])
(P/'DELIVERY.md').write_text('\n'.join(lines)+'\n')
files=[P/'RESULT.json',P/'DELIVERY.json',P/'DELIVERY.md',P/'continuation_launcher.py',P/'sla_resume_orchestrate.py',P/'sla_resume_finalize.py']
files+=list(Q.glob('*'))
files += [P/'SLA_RESUME_CONTRACT.json',P/'SLA_RESUME_CPU_CHECKS.json',P/'SLA_0531_SNAPSHOT.json']
files += list((P/'attempts').glob('sla_resume*'))
for a in new:
 for name in ['guard','checker','native','guard_log']:
  if a.get(name):files.append(Path(a[name]))
hashes={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.is_file()}
(P/'FILEHASHES.json').write_text(json.dumps(hashes,indent=2))
print(json.dumps(dict(scope=len(rows),statuses=counts,new_attempts=len(new),accepted_added=0)))

# Scope-limited continuation audit; no registry mutation.
ks=json.loads((P/'SLA_REMAINING.json').read_text())
selected=[r for r in rows if r['key'] in ks]
for r in selected:
 a=[a for a in r['attempts'] if 'sla_resume_20261027T0529' in a['guard']][-1:]
 if a and a[0].get('checker'):
  c=json.loads(Path(a[0]['checker']).read_text())
  r['original_checker_status']=c.get('status')
  r['numerical_failure_established']=False
  r['input_preservation_verified']=c.get('inputs_unchanged',None)
  r['remaining_reason']=c.get('sync_error') or c.get('error') or r['remaining_reason']
(P/'RESULT.json').write_text(json.dumps(rows,indent=2))
d['original_sla']='2026-10-27T05:31:00+00:00'
d['sla_snapshot']=str(P/'SLA_0531_SNAPSHOT.json')
d['sla_overdue']=True
d['continuation_exact_keys']=ks
d['continuation_results']=selected
d['remaining']=[dict(key=r['key'],status=r['status'],reason=r['remaining_reason']) for r in rows if not r['numerical_success']]
d['time_notes']='Guard durations use same-host timestamps. Remote wall clocks differ; CPU/device time splits are unknown. acquired_to_child_start includes admission checks and must not be called device execution.'
(P/'DELIVERY.json').write_text(json.dumps(d,indent=2))
for a in new:
 folder=Path(a['guard']).parent
 files += [x for x in folder.iterdir() if x.is_file()]
 for launch in folder.glob('launch_*'):
  if launch.is_dir():files += [x for x in launch.iterdir() if x.is_file()]
files += [P/'SLA_REMAINING.json',P/'REPLACEMENT_CLEARANCE.json',P/'SEED_DECLARATIONS.json',P/'DEPLOYMENT.md',P/'CONTINUATION_DEPLOYMENT.md']
hashes={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.is_file()}
(P/'FILEHASHES.json').write_text(json.dumps(hashes,indent=2))
