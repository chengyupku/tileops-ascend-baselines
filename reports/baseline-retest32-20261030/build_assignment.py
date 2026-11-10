import json,csv,hashlib,time,collections
from pathlib import Path
ROOT=Path('/root/dyq/workspace');P=ROOT/'reports/baseline-retest32-20261030'
amnesty_path=ROOT/'.cache/all-pairs/full_rollout_20261024/repairs/resume_20261025/boundary_1557/user_amnesty_0323/POLICY.json';amnesty={tuple(k) for k in json.loads(amnesty_path.read_text())['amnesty_keys']}
server2={'ops-math','ops-nn','tilelang','TileKernels','DeepSelect'};tasks=[]
for c in ['baseline-cache-20261027','baseline-old1076-20261029']:
 source=ROOT/'reports'/c/'local/CACHE.jsonl'
 for line_no,line in enumerate(source.open(),1):
  r=json.loads(line);assert r['status']=='cached' and r['performance_valid'] is True
  e=r['identity']['environment'];assert e['hostname']=='aa' and e['device_name']=='Ascend950DT_9592'
  t={'id':r['task_id'],'key':r['key'],'host':'server2' if r['key'][0] in server2 else 'server1','case':r['identity']['case'],'timeout':max(600,min(3600,int((r['finished_at']-r['started_at'])*4+120))),'eligible_basis':r.get('eligible_basis','local_fullchecker_cached'),'legacy_amnesty':tuple(r['key']) in amnesty,'local_cache_key':r['cache_key'],'local_source':{'file':str(source),'line':line_no,'correctness_receipt':r['correctness_receipt'],'guard_receipt':r['guard_receipt']},'local_measurement_seconds':r['finished_at']-r['started_at'],'expected_seconds':(r['finished_at']-r['started_at'])*1.15+12,'target_aicore_count':32,'target_vectorcore_count':64}
  tasks.append(t)
assert len(tasks)==2434 and len({tuple(t['key']) for t in tasks})==2434
for host in ['server1','server2']:
 selected=[t for t in tasks if t['host']==host];(P/(host+'_ASSIGNMENTS.json')).write_text(json.dumps({'host':host,'count':len(selected),'pairs':selected},ensure_ascii=False,indent=2))
(P/'PLAN.json').write_text(json.dumps({'scope':2434,'pairs':tasks},ensure_ascii=False,indent=2))
with (P/'ASSIGNMENTS.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['host','id','baseline','operator','case','legacy_amnesty','estimated_seconds'])
 for t in tasks:w.writerow([t['host'],t['id'],*t['key'],t['legacy_amnesty'],round(t['expected_seconds'],2)])
summary={'created_at':time.time(),'scope':2434,'source_audit':'reports/baseline-core-count-audit-20261030/SUMMARY.json','owners':{'server1':'User will assign another agent via HANDOFF.txt','server2':'Current conversation'},'hosts':{h:{'pairs':sum(t['host']==h for t in tasks),'predicted_hours':sum(t['expected_seconds'] for t in tasks if t['host']==h)/3600,'baseline_counts':dict(collections.Counter(t['key'][0] for t in tasks if t['host']==h))} for h in ['server1','server2']},'amnesty_pairs':sum(t['legacy_amnesty'] for t in tasks),'old1076_remaining103':'Remains paused; excluded from these2434 retests','terminal_states':['baseline支持且已缓存','baseline不支持'],'assignment_digest':hashlib.sha256((P/'PLAN.json').read_bytes()).hexdigest()};(P/'CAMPAIGN.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False))
