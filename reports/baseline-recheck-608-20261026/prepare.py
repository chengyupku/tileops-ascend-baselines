import pathlib,csv,json,collections,datetime
R=pathlib.Path(__file__).resolve().parent;W=R.parents[1]
raw=list(csv.DictReader(open(W/'reports/baseline-reconciled-20261026/CANDIDATES_TO_CONFIRM.csv',encoding='utf-8-sig')))
sample=json.load(open(W/'reports/baseline-sample-recheck-20261026/RESULT.json'))
skip={tuple(x['key']) for x in sample['results']};rows=[]
for x in raw:
 k=tuple(x[f] for f in ['baseline','operator','case_id'])
 if k not in skip:rows.append(dict(x,key=list(k)))
assert len(rows)==608 and len({tuple(x['key']) for x in rows})==608
families=collections.defaultdict(list)
for x in rows:families[(x['baseline'],x['operator'])].append(x)
bins=[[] for _ in range(8)]
for k,v in sorted(families.items(),key=lambda kv:(-len(kv[1]),kv[0])):
 b=min(range(8),key=lambda i:len(bins[i]));bins[b]+=v
now=datetime.datetime.now(datetime.timezone.utc);deadline=now+datetime.timedelta(hours=1)
plan={'started_at':now.isoformat(),'deadline':deadline.isoformat(),'scope':608,'requested_number':609,'reconciliation':'611 minus3 sampled=608 exact keys','workers':8,'effort':'medium','npu':False,'groups':[]}
for i,rs in enumerate(bins,1):
 p=R/f'worker{i:02}';p.mkdir(exist_ok=True)
 (p/'SCOPE.json').write_text(json.dumps({'rows':rs},ensure_ascii=False,indent=2))
 task=f"先阅读 {R}/WORKER_POLICY.txt，这是完整任务边界。\n你的排他scope={p}/SCOPE.json，共{len(rs)}项。只写{p}。最终截止{deadline.isoformat()}，请20分钟内交首版以便主agent复核。相邻worker在检查其他族，不得重复派工或扩scope。旧字段candidate_to_confirm不是结论。\n"
 (p/'TASK.txt').write_text(task)
 plan['groups'].append({'name':p.name,'directory':str(p),'task_file':str(p/'TASK.txt'),'count':len(rs),'families':dict(collections.Counter(x['baseline']+'/'+x['operator'] for x in rs))})
(R/'PLAN.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2));(R/'SCOPE608.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));print(json.dumps(plan,ensure_ascii=False,indent=2))
