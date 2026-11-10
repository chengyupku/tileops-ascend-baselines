import pathlib,json,datetime,sys
P=pathlib.Path(__file__).parent
rows=json.loads((P/'SCOPE.json').read_text())['rows']; specs=json.loads((P/'contracts.json').read_text())
def contract(r):
 s=specs[r['operator']]['spec']; matches=[w for w in s['workloads'] if r['case_id']==w['label'] or r['case_id'].startswith(w['label']+'-')]; w=max(matches,key=lambda w:len(w['label'])) if matches else None
 return {'signature':s['signature'],'workload':w,'case_id':r['case_id']}
def record(op,decision,reason,evidence,route,case=None):
 done=[json.loads(l)['key'] for l in (P/'decisions.jsonl').read_text().splitlines()] if (P/'decisions.jsonl').exists() else []
 for r in rows:
  if r['operator']!=op or (case and r['case_id']!=case) or r['key'] in done: continue
  c=contract(r); w=c['workload']; detail=json.dumps(w,ensure_ascii=False,separators=(',',':'))
  obj={'key':r['key'],'decision':decision,'reason':r['case_id']+'：'+reason+' 本项参数='+detail,'evidence':[str(pathlib.Path(e.split(':')[0]).resolve())+(':'+e.split(':',1)[1] if ':' in e else '') for e in evidence]+[specs[op]['file']+':'+op],'contract':c,'api_or_rejected_routes':route,'decided_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  with (P/'decisions.jsonl').open('a') as f: f.write(json.dumps(obj,ensure_ascii=False)+'\n')
 n=len((P/'decisions.jsonl').read_text().splitlines()); (P/'HEARTBEAT.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'current_family':op,'completed':n,'blockers':[]},ensure_ascii=False,indent=2)); print(op,n)
