import json,subprocess,concurrent.futures,collections
from pathlib import Path
R=Path('/root/dyq/workspace'); O=Path(__file__).resolve().parent; H=json.load(open(O/'HANDOFF.snapshot.json')); ids={e['id'] for e in H['entries']}
probe='''import json,pathlib
p=pathlib.Path(ROOT)/'reports/baseline-cache-20261027'/HOST
rows=[];checks={}
for line in (p/'journal.jsonl').read_text().splitlines():
 r=json.loads(line)
 if r['id'] not in IDS:continue
 rows.append(r)
 if str(r.get('repair_revision','')).startswith('support-'):
  c=pathlib.Path(r.get('measurement','')).parent/'correctness.json'
  if c.exists():
   z=json.loads(c.read_text())
   if z.get('status')=='verified':checks[r['id']]={'path':str(c),'status':'verified'}
cache=[]
for line in (p/'CACHE.jsonl').read_text().splitlines():
 r=json.loads(line)
 if r.get('task_id') in IDS and r.get('performance_valid') and r.get('status')=='cached':cache.append(r)
print(json.dumps({'host':HOST,'journal':rows,'checks':checks,'cache':cache}))
'''
def fetch(v):
 host,alias,root=v;code='ROOT='+repr(root)+'\nHOST='+repr(host)+'\nIDS='+repr(ids)+'\n'+probe
 cmd=['python3','-'] if not alias else ['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes',alias,'python3 -']
 s=subprocess.run(cmd,input=code,text=True,capture_output=True,check=True,timeout=60);data=json.loads(s.stdout);(O/(host+'-receipts.json')).write_text(s.stdout);return data
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:data=list(ex.map(fetch,[('local',None,str(R)),('server1','openlibing-DevEnv-661495','/opt/tileops-baseline-cache/workspace'),('server2','openlibing-DevEnv-055409','/opt/tileops-all-pairs/workspace')]))
decisions={e['id']:e['existing_decision'] for e in H['entries'] if e.get('existing_decision')};attempts=collections.defaultdict(set)
for d in data:
 for r in d['journal']:
  rev=str(r.get('repair_revision',''))
  if rev.startswith('support-'):attempts[r['id']].add(rev)
 for id,c in d['checks'].items(): decisions[id]={'id':id,'bucket':'baseline支持','evidence':c['path'],'host':d['host']}
 for c in d['cache']:decisions[c['task_id']]={'id':c['task_id'],'key':c['key'],'bucket':'baseline支持','host':d['host'],'cache_key':c.get('cache_key'),'evidence':str(O/(d['host']+'-receipts.json'))}
(O/'DECISIONS.json').write_text(json.dumps(decisions,ensure_ascii=False,indent=2));(O/'ATTEMPTS.json').write_text(json.dumps({k:sorted(v) for k,v in attempts.items()},indent=2))
print('decisions',collections.Counter(x['bucket'] for x in decisions.values()),'remaining',len(ids-decisions.keys()))
for e in H['entries']:
 cfg=e.get('last_queued_configuration')
 if e['id'] not in decisions and e['host']=='server1' and cfg and cfg['revision'] not in attempts[e['id']] and len(attempts[e['id']])<3:print('pilot',e['id'],e['key'],cfg);break
