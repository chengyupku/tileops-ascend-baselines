"""Read-only stdlib extraction of pipeline timing; run over SSH without device imports."""
import json,time
from pathlib import Path
h=Path('/opt/tileops-baseline-cache/workspace/reports/baseline-retest32-20261030/server1')
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return {}
rows=[];batches=[]
for b in sorted((h/'batches').iterdir()):
 plan=read(b/'plan.json');events=[]
 if (b/'events.jsonl').exists():
  for l in (b/'events.jsonl').read_text().splitlines():
   try:events.append(json.loads(l))
   except ValueError:pass
 batches.append({'batch':b.name,'events':events,'summary':read(b/'summary.json')})
 for j in plan.get('jobs',[]):
  o=Path(j['out']);m=read(o/'measurement.json');c=read(o/'correctness.json');g=read(b/j['id']/'guard.json');t=read(o/'task.json')
  rows.append({'batch':b.name,'id':j['id'],'key':j['key'],'out':str(o),'guard':{k:g.get(k) for k in ['requested_at','acquired_at','updated_at','phase','performance_valid','workload_exit_code']},'measurement':{k:m.get(k) for k in ['status','started_at','finished_at','end_to_end_mean_ms']},'correctness':{k:c.get(k) for k in ['status','started_at','finished_at','wall_seconds']},'phase':read(o/'phase.json'),'expected_seconds':t.get('expected_seconds'),'timing':{k:m.get('timing',{}).get(k) for k in ['latency_ms','cold_latency_ms']}})
print(json.dumps({'at':time.time(),'attempts':rows,'batches':batches,'journal':[json.loads(l) for l in (h/'journal.jsonl').read_text().splitlines()],'disk_status':read(h/'DISK_STATUS.json')}))
