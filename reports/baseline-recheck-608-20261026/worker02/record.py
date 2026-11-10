import json,datetime,pathlib,yaml
ROOT=pathlib.Path('/root/dyq/workspace'); OUT=ROOT/'reports/baseline-recheck-608-20261026/worker02'
ROWS=json.load(open(OUT/'SCOPE.json'))['rows']
SPECS={}
for p in (ROOT/'TileOPs/src/tileops/manifest/spec').glob('*.yaml'):
 for name,sp in (yaml.safe_load(p.read_text()) or {}).items():
  if isinstance(sp,dict): SPECS[name]=(p,sp)
def contract(row):
 p,sp=SPECS[row['operator']]
 matches=[w for w in sp.get('workloads',[]) if row['case_id']==w.get('label','') or row['case_id'].startswith(w.get('label','')+'-')]
 longest=max(len(w.get('label','')) for w in matches)
 matches=[w for w in matches if len(w.get('label',''))==longest]
 w=next((w for w in matches if str(w.get('out_dtype','')) and row['case_id'].endswith('-'+str(w['out_dtype']))),matches[0])
 suffix=row['case_id'][len(w['label'])+1:]
 return {'spec':str(p),'case_id':row['case_id'],'workload':w,'dtype_suffix':suffix,'signature':sp.get('signature',{})}
 raise ValueError(row)
def emit(baseline,op,decision,reason,evidence,route,select=lambda r:True):
 rows=[r for r in ROWS if r['baseline']==baseline and r['operator']==op and select(r)]
 with open(OUT/'decisions.jsonl','a') as f:
  for r in rows:
   c=contract(r)
   rec={'key':r['key'],'decision':decision,'reason':reason(r,c) if callable(reason) else r['case_id']+'：'+reason,'evidence':[str(ROOT/e) if not e.startswith('/') else e for e in evidence],'contract':c,'api_or_rejected_routes':route,'decided_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
   f.write(json.dumps(rec,ensure_ascii=False)+'\n')
 n=sum(1 for _ in open(OUT/'decisions.jsonl'))
 json.dump({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'current_family':baseline+'/'+op,'completed':n,'blockers':[]},open(OUT/'HEARTBEAT.json','w'),ensure_ascii=False,indent=2)
 print(op,len(rows),'total',n)
