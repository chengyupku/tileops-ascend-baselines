import json,subprocess,shlex,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[1];plan=json.loads((p/'PLAN.json').read_text());d=json.loads((p/'DECISIONS.json').read_text());src=json.loads((p/'SOURCE_ATTEMPTS.json').read_text());rows=json.loads((p/'RUN.snapshot.json').read_text())['journal'];actions={}
for host,alias,remote in [('local',None,str(root)),('server1','openlibing-DevEnv-661495','/opt/tileops-baseline-cache/workspace'),('server2','openlibing-DevEnv-055409','/opt/tileops-all-pairs/workspace')]:
 qp=p/f'THREEHOST_{host}_QUEUE.json';rel=qp.relative_to(root)
 if alias:q=json.loads(subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes',alias,'cat '+remote+'/'+str(rel)],capture_output=True,text=True,check=True).stdout)
 else:q=json.loads(qp.read_text())
 added=[]
 for t in plan['pairs']:
  tid=t['id']
  if t['host']!=host or tid in d or tid in q['pairs'] or any(x['id']==tid for x in rows):continue
  if host=='local' and t['key'][0]!='ops-nn':continue
  n=len(src.get(tid,[]))
  if n>=3:continue
  q['pairs'][tid]={'key':t['key'],'host':host,'entrypoint':'tools/baseline_old1076_nn_local_staging_v1.py' if host=='local' else 'tools/measure_baseline_cache_pair.py','revision':'support-old1076-nn-local-staging-v1' if host=='local' else 'support-old1076-threehost-original-env-v1','adaptation_attempt':n+1,'reason':'Original assigned-host deployment/candidate; unchanged native/compiler/checker; merged attempt ledger and no cached key','order':len(q['pairs'])};added.append(tid)
  if len(added)>=40:break
 if host=='local':
  tid='0934df1ff6a6fb958ac1';l=json.loads((p/'local/LIVE.json').read_text());v=q['pairs'][tid]
  if tid not in d and v['key']!=l.get('key') and v['entrypoint']=='tools/measure_baseline_cache_pair.py':
   hist={x['repair_revision'] for x in rows if x['id']==tid}|set(src.get(tid,[]));assert len(hist)<3
   v.update(entrypoint='tools/baseline_old1076_nn_local_staging_v1.py',revision='support-old1076-nn-local-staging-v1',adaptation_attempt=len(hist)+1,reason='Use original staged candidate to obtain full validation receipt; initial public registry admission is not a numeric verdict')
 data=json.dumps(q,indent=2);qp.with_name(qp.name+'.before0230-'+str(time.time_ns())).write_bytes(qp.read_bytes());tmp=qp.with_suffix('.tmp');tmp.write_text(data);tmp.replace(qp)
 if alias:
  code='import sys,pathlib;p=pathlib.Path('+repr(remote+'/'+str(rel))+');t=p.with_suffix(".tmp");t.write_text(sys.stdin.read());t.replace(p)'
  subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes',alias,'python3 -c '+shlex.quote(code)],input=data,text=True,check=True)
 actions[host]={'added':added,'queue_size':len(q['pairs'])}
(p/'REFILL_0230.json').write_text(json.dumps(actions,indent=2));print(json.dumps({h:{'added':len(a['added']),'total':a['queue_size']} for h,a in actions.items()}))
