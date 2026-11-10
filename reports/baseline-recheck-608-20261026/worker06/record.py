import json, pathlib, datetime, yaml
ROOT=pathlib.Path('/root/dyq/workspace'); OUT=ROOT/'reports/baseline-recheck-608-20261026/worker06'
rows=json.loads((OUT/'SCOPE.json').read_text())['rows']
spec={}
for p in (ROOT/'TileOPs/src/tileops/manifest/spec').glob('*.yaml'):
 for k,v in (yaml.safe_load(p.read_text()) or {}).items():
  if isinstance(v,dict):spec[k]=(p,v)
def record(baseline,op,decision,reason,evidence,route):
 p,s=spec[op]
 with (OUT/'decisions.jsonl').open('a') as f:
  for r in rows:
   if (r['baseline'],r['operator'])!=(baseline,op):continue
   found=[]
   for w in s.get('workloads',[]):
    if r['case_id']==w.get('label') or r['case_id'].startswith(w.get('label','NOTMATCH')+'-'):found.append(w)
   w=max(found,key=lambda x:len(x['label'])) if found else {}
   d={'key':r['key'],'decision':decision,'reason':r['case_id']+': '+reason,'evidence':[str(ROOT/e) if not e.startswith('/') else e for e in evidence]+[str(p)+':'+op],'contract':{'case':w,'signature':s.get('signature',{})},'api_or_rejected_routes':route,'decided_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
   f.write(json.dumps(d,ensure_ascii=False)+'\n')
 count=sum(1 for _ in (OUT/'decisions.jsonl').open())
 (OUT/'HEARTBEAT.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'family':[baseline,op],'completed':count,'blockers':[]},ensure_ascii=False))
 print(op,count)
