from pathlib import Path
import json,datetime
W=Path(__file__).resolve().parent
rows=[]
for g in sorted(W.glob('batch*/GUARD_RECEIPT.json')):
 r=json.loads(g.read_text());start=datetime.datetime.fromisoformat(r['start']).timestamp();cases=[]
 for f in W.glob('*/GUARD_RECEIPT.json'):
  if f==g or f.parent.name.startswith('batch'):continue
  c=json.loads(f.read_text())
  if c.get('batch_receipt')==g.parent.name:
   a=f.parent/'DEVICE_STARTED.json';z=f.parent/'DEVICE_FINISHED.json'
   if a.exists():cases.append(dict(case_id=f.parent.name,start=json.loads(a.read_text())['timestamp'],end=json.loads(z.read_text())['timestamp'] if z.exists() else None))
 first=min((c['start'] for c in cases),default=None);last=max((c['end'] for c in cases if c['end']),default=None)
 rows.append(dict(batch=g.parent.name,returncode=r['returncode'],guard_elapsed_seconds=r['elapsed_seconds'],queue_plus_launch_seconds=first-start if first else None,checker_span_seconds=last-first if first and last else None,cases=cases))
(W/'TIMING_AUDIT.json').write_text(json.dumps(dict(rows=rows,queue_plus_launch_seconds=sum(x['queue_plus_launch_seconds'] or 0 for x in rows),checker_span_seconds=sum(x['checker_span_seconds'] or 0 for x in rows),note='Host wall clocks. Queue_plus_launch includes shell/Python startup. Checker span includes CPU oracle, copies, allocations and all device work; not a latency benchmark.'),indent=2))
print('queue+launch',round(sum(x['queue_plus_launch_seconds'] or 0 for x in rows)), 'checker spans',round(sum(x['checker_span_seconds'] or 0 for x in rows)))
