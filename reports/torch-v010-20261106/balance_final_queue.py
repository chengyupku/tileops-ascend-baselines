"""Balance remaining work using measured representative cost; retain accepted ownership."""
import pathlib,json,tarfile,math,statistics
P=pathlib.Path(__file__).resolve().parent
rows=json.load(open(P/'QUEUE.json'))['tasks'];reps={}
def size(t):return sum(math.prod(x[0])*({'float32':4,'float64':8,'int64':8,'int32':4}.get(x[1],2)) for x in t['case']['tensors'].values())
for host in ('server1','server2'):
 for archive in ('representatives','expanded-representatives','final-representatives','gelu-representatives'):
  f=P/host/(archive+'.tar.gz')
  if not f.exists():continue
  with tarfile.open(f) as t:
   for m in t.getmembers():
    if not m.name.endswith('/measurement.json'):continue
    d=pathlib.PurePosixPath(m.name).parent
    measured=json.load(t.extractfile(m));task=json.load(t.extractfile(str(d/'task.json')));prep=json.load(t.extractfile(str(d/'prepare.json')))
    reps.setdefault(task['key'][1],[]).append((size(task),prep['cpu_seconds'],measured['timing']['latency_ms']))
def cost(t):
 samples=reps.get(t['key'][1],[])
 if not samples:return 60+size(t)/1e8
 return 30+statistics.median((cpu+18*ms/1000)*max(.1,size(t)/max(n,1)) for n,cpu,ms in samples)
accepted={};plans={h:[] for h in ('server1','server2')};load={h:0. for h in plans}
for h in plans:
 ledger=json.load(open(P/h/'PREACCEPTED.json'));accepted.update({tid:h for tid in ledger})
for r in sorted(rows,key=lambda r:cost(r['task']),reverse=True):
 t=dict(r['task']);t.update({k:r[k] for k in ('reasons','old_keys','source_contract')})
 h=accepted.get(t['id']) or min(plans,key=lambda h:(load[h],len(plans[h])))
 plans[h].append(t)
 if t['id'] not in accepted:load[h]+=cost(t)
for h,plan in plans.items():
 # Begin with six small unaccepted tasks to validate pipeline/collector turnover.
 remaining=sorted((t for t in plan if t['id'] not in accepted),key=cost);starter=remaining[:6];ids={t['id'] for t in starter}
 ordered=starter+[t for t in plan if t['id'] not in ids]
 (P/h/'PLAN.json').write_text(json.dumps({'rows':ordered},indent=2))
ids=[t['id'] for plan in plans.values() for t in plan];assert len(ids)==len(set(ids))==234
(P/'ASSIGNMENT.json').write_text(json.dumps({'hosts':{'server1':'openlibing-DevEnv-661495','server2':'openlibing-DevEnv-055409'},'total':234,'draft':False,'counts':{h:len(v) for h,v in plans.items()},'preaccepted':{h:sum(t['id'] in accepted for t in v) for h,v in plans.items()},'estimated_remaining_cost':load,'basis':'representative CPU preparation +18 device calls scaled by input bytes,30s fixed overhead; scheduling estimate, not completion promise; preaccepted pinned to actual host'},indent=2));print(load,{h:len(v) for h,v in plans.items()})
