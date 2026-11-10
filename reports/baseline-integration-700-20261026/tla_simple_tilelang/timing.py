from pathlib import Path
import json,datetime
P=Path(__file__).resolve().parent
seen={}
for f in (P/'runs').glob('*/GUARD_RUNNING.json'):
 r=json.loads(f.read_text());seen[r['pid']]=r
timing=[]
for pid,r in seen.items():
 a=datetime.datetime.fromisoformat(r['requested_at']);b=datetime.datetime.fromisoformat(r['acquired_at']);c=datetime.datetime.fromisoformat(r['updated_at'])
 timing.append(dict(pid=pid,requested_at=r['requested_at'],acquired_at=r['acquired_at'],queue_seconds=(b-a).total_seconds(),idle_gate_seconds=(c-b).total_seconds()))
started=json.loads((P/'STARTED.json').read_text())['utc'];now=datetime.datetime.now(datetime.timezone.utc)
result=dict(local_elapsed_seconds=(now-datetime.datetime.fromisoformat(started)).total_seconds(),unique_started_guards=timing,queue_seconds_sum=sum(x['queue_seconds'] for x in timing),note='No sum of repeated per-case batch queue times. Pending guards not represented until acquired. Compile durations unavailable.')
batches=[]
for f in P.glob('*.guard.json'):
 try:
  g=json.loads(f.read_text())
  if 'started' not in g or 'finished' not in g:continue
  matched=[r for r in seen.values() if g['started']-2<=datetime.datetime.fromisoformat(r['requested_at']).timestamp()<=g['finished']]
  # choose closest request to this launch, avoiding unrelated later requests.
  r=min(matched,key=lambda r:abs(datetime.datetime.fromisoformat(r['requested_at']).timestamp()-g['started'])) if matched else None
  acquired=datetime.datetime.fromisoformat(r['acquired_at']).timestamp() if r else None
  batches.append(dict(receipt=f.name,wall_seconds=g['finished']-g['started'],held_seconds=g['finished']-acquired if acquired else None,returncode=g.get('returncode')))
 except (ValueError,KeyError):pass
result['batch_timings']=batches
result['held_seconds_sum']=sum(b['held_seconds'] or 0 for b in batches)
r=json.loads((P/'RESULT.json').read_text());result['attempt_receipts']=len(r['attempts']);result['distinct_attempted_keys']=len({(x['baseline'],x['operator'],x['case_id']) for x in r['attempts']});result['additional_attempt_receipts']=result['attempt_receipts']-result['distinct_attempted_keys']
(P/'TIMINGS.json').write_text(json.dumps(result,indent=2))
