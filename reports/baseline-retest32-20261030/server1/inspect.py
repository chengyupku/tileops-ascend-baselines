#!/usr/bin/env python3
"""Read-only live collection; no runtime imports or NPU execution."""
import subprocess,json,time
from pathlib import Path
C=Path(__file__).resolve().parents[1]
script=r'''
import pathlib,json,subprocess,os,time
h=pathlib.Path('/opt/tileops-baseline-cache/workspace/reports/baseline-retest32-20261030/server1')
def read(p,d=None):return json.loads(p.read_text()) if p.exists() else d
state=read(h/'STATE.json',{});pid=state.get('identity',{}).get('pid');physical={}
if pid:
 try:
  fields=pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();physical={'pid':pid,'start_ticks':int(fields[19]),'state':fields[0],'matches':int(fields[19])==state['identity']['start_ticks']}
 except FileNotFoundError:physical={'pid':pid,'alive':False}
rows=[json.loads(l) for l in (h/'journal.jsonl').read_text().splitlines()] if (h/'journal.jsonl').exists() else []
latest={r['id']:r for r in rows}; cached=[r for r in latest.values() if r['status']=='cached']
active=read(h/'ACTIVE.json',{}); phases=[]
for j in active.get('jobs',[]):
 p=pathlib.Path(j['out']);phases.append({'id':j['id'],'phase':read(p/'phase.json',{}),'measurement_status':read(p/'measurement.json',{}).get('status')})
print(json.dumps({'checked_at':time.time(),'state':state,'process':physical,'active':active,'phases':phases,'cached':len(cached),'pending_review':[r for r in latest.values() if r['status']!='cached'],'remaining':1187-len(cached),'guard':json.loads(subprocess.run(['npu-guard','status','--json'],capture_output=True,text=True).stdout),'df':subprocess.check_output(['df','-B1',str(h)],text=True),'journal':rows,'cache':[json.loads(l) for l in (h/'CACHE.jsonl').read_text().splitlines()] if (h/'CACHE.jsonl').exists() else []}))
'''
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=script,text=True,capture_output=True,check=True)
x=json.loads(r.stdout);x['collected_local_at']=time.time();h=C/'server1';h.mkdir(exist_ok=True)
(h/'INSPECTION.latest.json').write_text(json.dumps(x,ensure_ascii=False,indent=2))
(h/'STATE.json').write_text(json.dumps({**x['state'],'inspection_cached':x['cached'],'inspection_remaining':x['remaining'],'checked_at':x['checked_at'],'session':json.loads((h/'TAKEOVER.json').read_text()).get('session')},indent=2))
(h/'CACHE.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in x['cache']))
(h/'journal.snapshot.json').write_text(json.dumps(x['journal'],ensure_ascii=False,indent=2))
support={r['id']:{'key':r['key'],'decision':'baseline支持且已缓存','evidence':r['measurement'],'guard':r['guard'],'host':'server1','aicore_count':32} for r in x['journal'] if r['status']=='cached'}
(h/'DECISIONS.json').write_text(json.dumps(support,ensure_ascii=False,indent=2))
import csv
with (h/'CLASSIFICATION.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['id','baseline','op','case','decision','evidence'])
 for tid,r in support.items():w.writerow([tid,*r['key'],r['decision'],r['evidence']])
summary={k:x[k] for k in ['checked_at','cached','remaining','process','df']}; summary['pending_review_count']=len(x['pending_review']);summary['npu_state']=x['guard']['state'];summary['utilization']=x['guard'].get('utilization')
(h/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False))
