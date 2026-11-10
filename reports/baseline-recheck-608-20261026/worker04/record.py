import json,pathlib,datetime,yaml
ROOT=pathlib.Path('/root/dyq/workspace'); OUT=ROOT/'reports/baseline-recheck-608-20261026/worker04'
ROWS=json.loads((OUT/'SCOPE.json').read_text())['rows']
SPECS={}
for p in (ROOT/'TileOPs/src/tileops/manifest/spec').glob('*.yaml'):
 for k,v in (yaml.safe_load(p.read_text()) or {}).items():
  if isinstance(v,dict): SPECS[k]=(p,v)
def record(op,decision,reason,evidence,route,baseline=None,predicate=lambda r:True):
 for r in ROWS:
  if r['operator']!=op or (baseline and r['baseline']!=baseline) or not predicate(r):continue
  p,s=SPECS[op]; ws=[w for w in s.get('workloads',[]) if r['case_id'].startswith(w.get('label','')+'-')]; w=max(ws,key=lambda w:len(w['label'])) if ws else {}
  contract={'case_id':r['case_id'],'workload':w,'signature':s.get('signature',{})}
  item={'key':r['key'],'decision':decision,'reason':f"{r['case_id']}: "+reason,'evidence':[str(ROOT/e) if not e.startswith('/') else e for e in evidence]+[str(p)+':'+op],'contract':contract,'api_or_rejected_routes':route,'decided_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  with (OUT/'decisions.jsonl').open('a') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
 count=sum(1 for _ in (OUT/'decisions.jsonl').open())
 (OUT/'HEARTBEAT.json').write_text(json.dumps({'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'current_family':op,'completed':count,'blocking':None},ensure_ascii=False))
