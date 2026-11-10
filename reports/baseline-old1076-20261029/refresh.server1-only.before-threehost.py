import csv,json,subprocess,collections,time
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[1]
code='''import json,pathlib
p=pathlib.Path('/opt/tileops-baseline-cache/workspace/reports/baseline-old1076-20261029/server1')
rows=[]
for line in (p/'journal.jsonl').read_text().splitlines():
 r=json.loads(line);c=pathlib.Path(r['measurement']).parent/'correctness.json';r['checker']=json.loads(c.read_text()) if c.exists() else {};rows.append(r)
print(json.dumps({'journal':rows,'live':json.loads((p/'LIVE.json').read_text())}))
'''
x=json.loads(subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,capture_output=True,check=True,timeout=20).stdout)
(O/'RUN.snapshot.json').write_text(json.dumps(x,ensure_ascii=False,indent=2))
d=json.load(open(O/'DECISIONS.json'));h=json.load(open(O/'HANDOFF.snapshot.json'))
for r in x['journal']:
 if r['checker'].get('status')=='verified':d[r['id']]={'id':r['id'],'key':r['key'],'bucket':'baseline支持','evidence':r['measurement'],'cache':r['status']=='cached'}
(O/'DECISIONS.json').write_text(json.dumps(d,ensure_ascii=False,indent=2))
with (O/'CLASSIFICATION.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['id','baseline','算子','case','分类','证据'])
 for e in h['entries']:
  if e['id'] in d:
   r=d[e['id']];w.writerow([e['id'],*e['key'],r['bucket'],r.get('evidence',r.get('correctness_receipt',''))])
s={'at':time.time(),'scope':1076,'classified':len(d),'buckets':dict(collections.Counter(r['bucket'] for r in d.values())),'remaining':1076-len(d),'live':x['live'],'csv':'reports/baseline-old1076-20261029/CLASSIFICATION.csv','note':'Unclassified tasks remain pending, never mapped automatically to unsupported.'}
(O/'SUMMARY.json').write_text(json.dumps(s,ensure_ascii=False,indent=2))
p=R/'.agent_memory/THREADS/baseline-old1076-20261029/STATE.json';state=json.load(open(p));state.update(phase=x['live']['phase'],latest_summary=s);p.write_text(json.dumps(state,ensure_ascii=False,indent=2))
print(json.dumps({k:s[k] for k in ('classified','buckets','remaining')},ensure_ascii=False))
