import json,pathlib,runpy,subprocess,time
P=pathlib.Path(__file__).resolve().parent
R=P.parents[1]
def read(f):return json.loads((P/f).read_text())
def put(f,d):
 p=P/f;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,ensure_ascii=False));t.replace(p)
snap=read('RUN.snapshot.json');pilot=[x for x in snap['journal'] if x['id']=='61f8342ba0530d2b7931' and x.get('repair_revision')=='support-old1076-nn-arm-complete-v3'];assert pilot and pilot[-1]['status']=='cached'
assert pilot[-1]['checker'].get('status')=='verified',pilot[-1]['checker'].keys()
q=read('THREEHOST_local_QUEUE.json');decisions=read('DECISIONS.json');history=read('SOURCE_ATTEMPTS.json');plan=read('PLAN.json');assignment=read('THREEHOST_ASSIGNMENTS.json');byid={t['id']:t for t in plan['pairs']}
rows=[json.loads(x) for x in (P/'local/journal.jsonl').read_text().splitlines()];seen={x['id'] for x in snap['journal']}|{x['id'] for x in rows}|{json.loads(x)['task_id'] for x in (P/'local/CACHE.jsonl').read_text().splitlines()}
active=read('local/LIVE.json').get('key');ops=set(read('ARM_NN_FRESH_RECEIPT.json')['ops']);routes=runpy.run_path(str(R/'TileOPs/benchmarks/ascend_all_pairs/ops_nn/catalog.py'))['ROUTES'];selected={}
for k,v in q['pairs'].items():
 route=routes.get(v['key'][1]);apis=set(route[0].split(' + ')) if route else set()
 if k in seen or k in decisions or history.get(k) or v['key']==active or v['key'][0]!='ops-nn' or not apis or not apis<=ops:continue
 assert byid[k]['host']=='local';selected[k]=v
assert len(selected)>=10,len(selected)
put('LOCAL_QUEUE.before-ready-balance.json',q)
for k in selected:q['pairs'].pop(k)
put('THREEHOST_local_QUEUE.json',q)
# Remove eligibility first; then verify no in-flight local execution claimed a key.
active=read('local/LIVE.json').get('key');rows=[json.loads(x) for x in (P/'local/journal.jsonl').read_text().splitlines()];started={x['id'] for x in rows}
for k,v in list(selected.items()):
 if k in started or v['key']==active:q['pairs'][k]=v;selected.pop(k)
put('THREEHOST_local_QUEUE.json',q)
queues={'server1':{'pairs':{}},'server2':{'pairs':{}}}
for i,(k,v) in enumerate(selected.items()):
 h=['server1','server2'][i%2];e=dict(v,host=h,entrypoint='tools/baseline_old1076_nn_arm_staging_v3.py',revision='support-old1076-nn-arm-balanced-v1',adaptation_attempt=1,reason='Unattempted uncached exact key uniquely migrated from local after validated original58API ARM deployment pilot; original staged interface/kernel/compiler/checker retained')
 queues[h]['pairs'][k]=e;byid[k]['host']=h;assignment['assignments'][k]['host']=h
put('PLAN.before-ready-balance.json',read('PLAN.json'));put('PLAN.json',plan);put('THREEHOST_ASSIGNMENTS.json',assignment)
for h,alias,root in [('server1','openlibing-DevEnv-661495','/opt/tileops-baseline-cache/workspace'),('server2','openlibing-DevEnv-055409','/opt/tileops-all-pairs/workspace')]:
 code='''from pathlib import Path
import json
p=Path(ROOT)/'reports/baseline-old1076-20261029'
def put(f,d):
 t=f.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(f)
f=p/'PLAN.json';d=json.loads(f.read_text())
for t in d['pairs']:
 if t['id'] in QUEUE['pairs']:t['host']=HOST
put(f,d);f=p/('NN_BALANCED_'+HOST+'_QUEUE.json');assert not f.exists();put(f,QUEUE)
print(len(QUEUE['pairs']))
'''
 subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes',alias,'python3 -'],input='ROOT='+repr(root)+'\nHOST='+repr(h)+'\nQUEUE='+repr(queues[h])+'\n'+code,text=True,check=True)
 put('NN_BALANCED_'+h+'_QUEUE.json',queues[h])
put('NN_BALANCE_COMMITTED.json',{'at':time.time(),'counts':{h:len(q['pairs']) for h,q in queues.items()},'migrated':list(selected),'pilot':'61f8342ba0530d2b7931','remaining_local_queue_entries':len(q['pairs']),'note':'Cached/local in-flight excluded; original historical attempts preserved; queues ready, runners require actual verification'})
print('migrated',len(selected))
