import ast, collections, datetime as dt, json, os
from pathlib import Path
B=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026'); O=B/'hour_guard_queue_audit_0317'
LO=dt.datetime.fromisoformat('2026-10-27T02:17:42+00:00').timestamp(); HI=LO+3600
TASKS=['threehost_linear49','threehost_nn11','threehost_pto_original23']
def read(p):
 try:return json.loads(p.read_text())
 except:return {}
def ep(v):
 if isinstance(v,(int,float)):return v
 if isinstance(v,str):
  try:return dt.datetime.fromisoformat(v).timestamp()
  except:return None
def iso(v):return dt.datetime.fromtimestamp(v,dt.timezone.utc).isoformat() if v is not None else None
def slim(m):return {k:m[k] for k in ['owner','phase','requested_at','acquired_at','updated_at','child','external_pids','exit_code','failure_snapshot'] if k in m}
rows=[];excluded=[]
for task in TASKS:
 for root,dirs,files in os.walk(B/task):
  dirs[:]=[d for d in dirs if not any(x in d for x in ['CACHE','cache','package','snapshot','ASCEND','native','vendors'])]
  for name in files:
   if not (name=='GUARD_RESULT.json' or name.endswith('.guard.json')):continue
   p=Path(root)/name;d=read(p);s=ep(d.get('started_epoch',d.get('started',d.get('requested_at'))));e=ep(d.get('finished_epoch',d.get('finished',d.get('finished_at'))))
   if s is None or e is None:excluded.append({'path':str(p.relative_to(B)),'reason':'missing explicit request/end timestamps; not infer from blocker metadata'});continue
   if e<LO or s>HI:continue
   rel=str(p.relative_to(B));host=d.get('host') or ('server1' if '/server1' in rel else 'server2' if '/server2' in rel else 'local')
   r={'path':rel,'task':task,'host':host,'start':iso(s),'end':iso(e),'wall_s':e-s,'window_overlap_s':max(0,min(e,HI)-max(s,LO)),'returncode':d.get('returncode'),'request_clock':'local/controller' if name.endswith('.guard.json') else 'execution host','metadata':[],'logs':[],'own_requested_at':None,'own_acquired_at':None,'child_started_at':None,'classification':'unknown'}
   ms=[(p,d.get('metadata',{}))]+[(p.parent/n,read(p.parent/n)) for n in ['GUARD_METADATA.json','GUARD_RUNNING.json','GUARD_FINAL.json'] if (p.parent/n).exists()]
   for mp,m in ms:
    if not m:continue
    own=m.get('owner')=='baseline700-20261026/'+task
    r['metadata'].append({'path':str(mp.relative_to(B)),'belongs_to_requester':own,'fields':slim(m)})
    if own:
     r['own_requested_at']=m.get('requested_at',r['own_requested_at']);r['own_acquired_at']=m.get('acquired_at',r['own_acquired_at'])
     if m.get('phase')=='running' and m.get('child'):r['child_started_at']=m.get('updated_at')
   lps=[p.with_suffix('.log')] if name.endswith('.guard.json') else [p.parent/'guard.log',p.parent/'stderr.log']
   for lp in lps:
    if not lp.exists():continue
    ls=[]; seen=set()
    for i,line in enumerate(lp.read_text(errors='replace').splitlines(),1):
     if '[npu-guard]' not in line:continue
     if line in seen:continue
     seen.add(line);ls.append({'line':i,'text':line})
    r['logs'].append({'path':str(lp.relative_to(B)),'selected_unique_guard_lines':ls})
   txt='\n'.join(x['text'] for l in r['logs'] for x in l['selected_unique_guard_lines'])
   if d.get('returncode')==75:
    if r['own_acquired_at']:
     r['classification']='flock_queue_then_physical_idle_confirmation_deadline'
    elif 'waiting for reservation' in txt:r['classification']='flock_or_fifo_queue_no_launch'
    else:r['classification']='wait_timeout_phase_unknown'
   elif d.get('returncode')==76 and 'external' in txt:r['classification']='launched_then_external_context_contamination'
   elif 'acquired device' in txt:r['classification']='launched_cpu_device_split_unknown'
   if r['own_requested_at'] and r['own_acquired_at']:r['flock_fifo_wait_s']=ep(r['own_acquired_at'])-ep(r['own_requested_at'])
   if r['child_started_at'] and r['own_acquired_at']:r['post_acquire_pre_child_s']=ep(r['child_started_at'])-ep(r['own_acquired_at'])
   rows.append(r)
# Keep duplicate timestamps/host/task only once, retaining provenance references.
uniq={}
for r in rows:
 k=(r['task'],r['host'],r['start'],r['end'])
 if k in uniq:uniq[k].setdefault('duplicate_receipts',[]).append(r['path'])
 else:uniq[k]=r
rows=list(uniq.values());rows.sort(key=lambda r:r['start'])
stats={}
for host in ['local','server1','server2']:
 rr=[r for r in rows if r['host']==host];bad=[r for r in rr if r['returncode']==75]
 stats[host]={'sampled_guard_invocations':len(rr),'returncodes':dict(collections.Counter(str(r['returncode']) for r in rr)),'rc75_invocation_wait_sum_s':sum(r['window_overlap_s'] for r in bad),'note':'sum of caller wall time; overlapping waiters are not device downtime; rc0 is not numerical success'}
(O/'extracted_guard_evidence.json').write_text(json.dumps({'window':[iso(LO),iso(HI)],'rows':rows,'stats':stats,'excluded_missing_time':excluded},ensure_ascii=False,indent=2))
print(json.dumps({'stats':stats,'count':len(rows),'excluded_missing_time':excluded},ensure_ascii=False,indent=2))
