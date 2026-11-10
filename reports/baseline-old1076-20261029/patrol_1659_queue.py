import json,time
from pathlib import Path
p=Path('/opt/tileops-baseline-cache/workspace/reports/baseline-old1076-20261029')
qpath=p/'TRANSFORMER_REMAINING_QUEUE.json';q=json.loads(qpath.read_text());plan=json.loads((p/'PLAN.json').read_text());source=json.loads((p/'SOURCE_ATTEMPTS.json').read_text());rows=[json.loads(s) for s in (p/'server1/journal.jsonl').read_text().splitlines()];live=json.loads((p/'server1/LIVE.json').read_text())
ids=['d181585ae57c562ff0c3','f17a5738fb5dde5ae383','1598d5bf77e049712d62','fbdff2c3577cd5dd16c0']
for tid in ids:
 item=q['pairs'][tid];history=[r for r in rows if r['id']==tid]
 assert history and all(r['status']!='cached' for r in history)
 assert item['key']!=live.get('key')
 revisions=set(source.get(tid,[]))|{r['repair_revision'] for r in history if r.get('repair_revision','').startswith('support-')}
 assert len(revisions)==1,(tid,revisions)
 item['adaptation_attempt']=2
 if tid==ids[0]:
  item.update(entrypoint='tools/baseline_old1076_nn_arm_v2.py',revision='support-old1076-nn-original-arm-v2',reason='Import original BMM module directly under private package namespace; omit unrelated historical registration side effects pointing to absent local path; unchanged vendor/native/checker')
 elif tid==ids[-1]:
  item.update(entrypoint='tools/baseline_old1076_transformer_sequence_v3.py',revision='support-old1076-transformer-sequence-v3',reason='Observed workspace OOM inside original native call; release completed API workspace using original synchronize+cleanup after each complete independent sequence, preserve original kernel/checker, bounded1800seconds')
 else:
  item.update(revision='support-old1076-transformer-reset-budget-v3',reason='Checker incomplete at600second cap; bounded1800second retry with unchanged original kernel/checker and existing reset cleanup')
 for t in plan['pairs']:
  if t['id']==tid and tid!=ids[0]:t['timeout']=1800
for path,data in [(p/'PLAN.json',plan),(qpath,q)]:
 backup=path.with_name(path.name+'.before-patrol1659-'+str(time.time_ns()));backup.write_bytes(path.read_bytes());tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)
print(json.dumps({'updated':ids,'live':live,'count':len(q['pairs'])}))
