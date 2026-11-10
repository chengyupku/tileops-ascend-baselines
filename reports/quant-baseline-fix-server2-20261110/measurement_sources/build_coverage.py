"""Separate measured failures from unmeasured admission exclusions."""
import pathlib,json,collections
P=pathlib.Path(__file__).resolve().parent
rows=json.loads((P/'RESULTS.json').read_text());tasks=json.loads((P/'TASKS.json').read_text())
jobs=json.loads((P/'FORMAL_JOBS.pending.json').read_text())+json.loads((P/'OPS_MATH_FORMAL_PLAN.json').read_text())['jobs']
gate={tuple(x) for x in json.loads((P/'FORMAL_GATE.json').read_text())['allowed_op_variants']}
progress=[]
for path in P.glob('SEPARATED*PROGRESS.json'):
 progress.extend(json.loads(path.read_text())['records'])
explicit={}
timeouts={}
for path in (P/'remote').glob('batch*.progress.json'):
 for record in json.loads(path.read_text()):
  if record.get('exit_code')==124:
   j=record['job'];timeouts.setdefault((j['index'],j['variant']),[]).append({'attempt':j['attempt'],'batch':path.name,'seconds':record.get('seconds')})
for x in progress:
 if 'job' in x:explicit[(x['job']['index'],x['job']['variant'])]=x
out=[]
for j in jobs:
 i,v=j['index'],j['variant'];matching=[x for x in rows if x['index']==i and x['variant']==v];accepted=[x for x in matching if x['status']=='formal_raw_verified'];failed=[x for x in matching if x['status']=='failed'];reason=None
 if accepted:status='formal_verified'
 elif (i,v) in timeouts:status='measured_timeout_not_registered';reason=timeouts[(i,v)]
 elif failed:status='measured_attempt_failed_not_registered';reason=[{'attempt':x['attempt'],'phase':x.get('phase'),'error':x.get('error')} for x in failed]
 elif (i,v) in explicit:status='unmeasured_deferred';reason=explicit[(i,v)].get('reason',explicit[(i,v)]['status'])
 elif (tasks[i]['op'],v) not in gate and v!='ops_math':status='unmeasured_not_admitted';reason='No valid family pilot or concrete environment/compiler/correctness limitation; not an exact-case numerical failure.'
 else:status='pending'
 out.append({'index':i,'op':tasks[i]['op'],'case':tasks[i]['case'],'variant':v,'status':status,'reason':reason,'attempts':[{k:x[k] for k in ('attempt','status','result_sha256')} for x in matching]})
(P/'COVERAGE.json').write_text(json.dumps(out,indent=2))
print(dict(collections.Counter(x['status'] for x in out)))
