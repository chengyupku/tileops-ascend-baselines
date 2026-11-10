import json,time,subprocess,shlex
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[1];plan=json.loads((p/'PLAN.json').read_text());d=json.loads((p/'DECISIONS.json').read_text());rows=json.loads((p/'RUN.snapshot.json').read_text())['journal'];seen={x['id'] for x in rows};src=json.loads((p/'SOURCE_ATTEMPTS.json').read_text());qs={h:{'pairs':{}} for h in ['server1','server2']};moved=[]
for t in plan['pairs']:
 tid=t['id']
 if tid in d or tid in seen or t['key'][0]!='ops-transformer' or t['key'][1] not in ['GroupedGemmFwdOp','MoEGroupedGemmFwdOp']:continue
 if t['case']['params'].get('activation') is not None:continue
 assert tid not in json.loads((p/'THREEHOST_local_QUEUE.json').read_text())['pairs']
 n=len(src.get(tid,[]));assert n<3
 host=['server1','server2'][len(moved)%2];t['host']=host;qs[host]['pairs'][tid]={'key':t['key'],'host':host,'entrypoint':'tools/baseline_old1076_transformer_arm_graph_v2.py','revision':'support-old1076-transformer-arm-grouped-v2','adaptation_attempt':n+1,'reason':'Original ARM GraphCall vendor proven BMM fullchecker/cache; independent original grouped API adapter, fullchecker required'};moved.append((tid,host))
(p/'PLAN.json').write_text(json.dumps(plan,indent=2));a=json.loads((p/'THREEHOST_ASSIGNMENTS.json').read_text())
for tid,h in moved:a['assignments'][tid]['host']=h
(p/'THREEHOST_ASSIGNMENTS.json').write_text(json.dumps(a,indent=2))
for h,q in qs.items():(p/f'ARM_GRAPH_{h}_QUEUE.json').write_text(json.dumps(q,indent=2))
(p/'ARM_GRAPH_BALANCE.json').write_text(json.dumps({'at':time.time(),'moved':moved,'pilot':'e61d710e04ac06f92492 cached','status':'deploying'},indent=2));print(moved)
