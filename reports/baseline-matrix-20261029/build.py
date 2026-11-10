import csv,json,gzip,subprocess,datetime,hashlib
from pathlib import Path
from collections import defaultdict
ROOT=Path('/root/dyq/workspace'); OUT=Path(__file__).resolve().parent
CAT=ROOT/'.worktrees/TileOPs-baseline-catalog/docs/ascend950/baselines'
CAM=ROOT/'reports/baseline-cache-20261027'
cases=defaultdict(set)
for r in json.loads((CAT/'cases.json').read_text())['cases']: cases[r['op']].add(r['case_id'])
libs=[r['baseline'] for r in csv.DictReader((CAT/'libraries.csv').open())]
excluded=set()
for line in gzip.open(CAT/'observed_pairs.jsonl.gz','rt'):
 r=json.loads(line)
 if r['status']=='excluded': excluded.add(tuple(r['key']))
probe='''import json,pathlib
p=pathlib.Path(CAMPAIGN)
cache=[]
for line in (p/HOST/'CACHE.jsonl').read_text().splitlines():
 try:r=json.loads(line)
 except ValueError:continue
 if r.get('status')=='cached' and r.get('performance_valid') is True:cache.append(r['key'])
triage=p/'support-run-20261029'/(HOST+'-TRIAGE.json')
t=json.loads(triage.read_text()) if triage.exists() else {}
print(json.dumps({'cached':cache,'triage':t},ensure_ascii=False))
'''
cache=set(); snapshots={}
for host,alias,root in [('local',None,str(ROOT)),('server1','openlibing-DevEnv-661495','/opt/tileops-baseline-cache/workspace'),('server2','openlibing-DevEnv-055409','/opt/tileops-all-pairs/workspace')]:
 code='CAMPAIGN='+repr(root+'/reports/baseline-cache-20261027')+'\nHOST='+repr(host)+'\n'+probe
 cmd=['python3','-c',code] if not alias else ['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','-o','ConnectTimeout=10',alias,'python3 -']
 run=subprocess.run(cmd,input=code if alias else None,text=True,capture_output=True,timeout=60,check=True)
 data=json.loads(run.stdout); (OUT/(host+'-snapshot.json')).write_text(json.dumps(data,ensure_ascii=False))
 cache.update(map(tuple,data['cached'])); snapshots[host]={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cached':len(set(map(tuple,data['cached'])))}
 def walk(x):
  if isinstance(x,dict):
   if x.get('bucket')=='baseline不支持' and isinstance(x.get('key'),list): excluded.add(tuple(x['key']))
   for v in x.values():walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 walk(data['triage'])
# Preserve conflicting evidence instead of silently redefining support.
conflicts=cache & excluded
with (OUT/'baseline_matrix.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['算子']+libs)
 for op,ids in sorted(cases.items()):
  cells=[]
  for lib in libs:
   keys={(lib,op,c) for c in ids};cells.append(f'{len(keys-excluded)}/{len(keys)}/{len(keys & cache)}')
  w.writerow([op]+cells)
meta={'cell':'命中case数/总case数/已缓存case数','hit_definition':'全部manifest case减去目录excluded及最新TRIAGE明确baseline不支持；未观测、待验证、一般报错仍计命中','cache_definition':'三机CACHE.jsonl中status=cached且performance_valid=true的key并集，按baseline/op/case去重；未重新审计源码有效性','operators':len(cases),'baselines':len(libs),'manifest_cases':sum(map(len,cases.values())),'cached_unique':len(cache),'snapshots':snapshots,'cache_excluded_conflicts':[list(k) for k in sorted(conflicts)]}
(OUT/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(meta,ensure_ascii=False))
