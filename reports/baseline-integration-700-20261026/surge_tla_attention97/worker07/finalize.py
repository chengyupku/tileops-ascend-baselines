from pathlib import Path
import json,datetime,ast,hashlib
P=Path(__file__).resolve().parent
rows=json.loads((P/'RESULT.json').read_text());attempts=[]
for prefix in ['local_runs','server2_evidence/runs','server1_evidence/runs']:
 for f in (P/prefix).glob('*/guard.json'):
  d=json.loads(f.read_text()); parent=f.parent;meta=parent/'GUARD_RUNNING.json';m=json.loads(meta.read_text()) if meta.exists() else {}
  a=dict(key=d['key'],host=d['host'],argv=d['argv'],requested_at=d['requested_at'],acquired_at=m.get('acquired_at'),child_start=m.get('updated_at'),ended_at=d['ended_at'],guard_exit=d['returncode'],guard_receipt=str(f),log=str(parent/'guard.log'),wall_seconds=d['end']-d['start'],device_seconds=None,held_cpu_seconds=None,child_identity=m.get('child'),checker=str(parent/'checker.json') if (parent/'checker.json').exists() else None)
  attempts.append(a)
for r in rows:
 own=[a for a in attempts if a['key']==r['key']]
 if own:r['attempts']=own
 if r['status']=='pending':
  r.update(status='individually_unverified_resource_blocked' if own else 'individually_unverified_not_submitted',reason='Three compatible hosts checked: local external NPU contexts and cooperative queue; server2 held by sibling work; server1 failed closed on incomplete /proc FD visibility (78). No original checker execution in this task. No representative result substituted.',guard=[a['guard_receipt'] for a in own] or None,checker=None,numerical_failure_established=False)
(P/'RESULT.json').write_text(json.dumps(rows,indent=2))
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
summary=dict(utc=now,task_type='actual integration / supplemental verification',keys=13,accepted_added=0,verified_added=0,device_child_started=sum(a['child_identity'] is not None for a in attempts),guard_attempts=len(attempts),guard_wall_seconds=sum(a['wall_seconds'] for a in attempts),device_seconds=None,held_cpu_seconds=None,status_counts={s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},attempts=attempts,remaining='9 exact keys still require original numerical verification; 3 original API blockers; 1 historical exact numerical failure retained',ready=False,checks=['13 unique exact keys preserved','Private Python syntax parses','Original source and ELF CPU identities recorded for 3 hosts','No kernel/compiler/checker/seed/tolerance changes','No accepted keys rerun; no registry/memory changes; no subworkers'],issues=['No original checker result: local/server2 reservation expiry75; server1 proc visibility failure78; no child started','CPU prepared deployment is not runtime-ready or accepted','Remote clock offset about -93 seconds; do not compare raw timestamps across hosts'])
(P/'DELIVERY.json').write_text(json.dumps(summary,indent=2))
(P/'HEARTBEAT.json').write_text(json.dumps(dict(utc=now,phase='completed_resource_blocked',current_key=None,last_substantive_output='RESULT.json and DELIVERY.json with all three attempts retained',blocker='local/server2 queue expiry75; server1 proc visibility78; no child started',next='parent review; schedule prepared runner when compatible host reservation can start'),indent=2))
for f in P.glob('*.py'):ast.parse(f.read_text())
(P/'FILEHASHES.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in P.iterdir() if f.is_file() and f.name not in ['FILEHASHES.json','HEARTBEAT.json']},indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='attempts'},indent=2))
