import csv,json,subprocess,collections,time
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[1]
def collect(host,root,alias=None):
 code="""import json,pathlib
p=pathlib.Path(ROOT)/'reports/baseline-old1076-20261029'/HOST
rows=[]
if (p/'journal.jsonl').exists():
 for line in (p/'journal.jsonl').read_text().splitlines():
  r=json.loads(line);c=pathlib.Path(r['measurement']).parent/'correctness.json';r['checker']=json.loads(c.read_text()) if c.exists() else {};r['host']=HOST
  if r.get('measurement_status')=='correctness_verified_uncached' and r['checker'].get('status')=='verified':
   r['reported_status']=r['status'];r['status']='correctness_verified_uncached'
  rows.append(r)
print(json.dumps({'journal':rows,'live':json.loads((p/'LIVE.json').read_text()) if (p/'LIVE.json').exists() else {}}))
""".replace('ROOT',repr(root)).replace('HOST',repr(host))
 cmd=['python3','-'] if alias is None else ['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes',alias,'python3 -']
 return json.loads(subprocess.run(cmd,input=code,text=True,capture_output=True,check=True,timeout=30).stdout)
from concurrent.futures import ThreadPoolExecutor
hosts=[('local',str(R),None),('server1','/opt/tileops-baseline-cache/workspace','openlibing-DevEnv-661495'),('server2','/opt/tileops-all-pairs/workspace','openlibing-DevEnv-055409')]
with ThreadPoolExecutor(max_workers=3) as pool:
 results=list(pool.map(lambda a:collect(*a),hosts))
x={'journal':[row for result in results for row in result['journal']],'live':results[1]['live'],'hosts':{host[0]:result['live'] for host,result in zip(hosts,results)}}
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
s={'at':time.time(),'scope':1076,'classified':len(d),'buckets':dict(collections.Counter(r['bucket'] for r in d.values())),'remaining':1076-len(d),'live':x['live'],'hosts':x['hosts'],'csv':'reports/baseline-old1076-20261029/CLASSIFICATION.csv','note':'Unclassified tasks remain pending, never mapped automatically to unsupported.'}
(O/'SUMMARY.json').write_text(json.dumps(s,ensure_ascii=False,indent=2))
p=R/'.agent_memory/THREADS/baseline-old1076-20261029/STATE.json';state=json.load(open(p));state.update(phase='running' if any(v.get('phase')=='running' for v in x['hosts'].values()) else 'queues_terminal',latest_summary=s);p.write_text(json.dumps(state,ensure_ascii=False,indent=2))
print(json.dumps({k:s[k] for k in ('classified','buckets','remaining')},ensure_ascii=False))
