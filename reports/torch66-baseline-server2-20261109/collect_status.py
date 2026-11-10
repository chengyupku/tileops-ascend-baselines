import pathlib,json,hashlib,math,collections
P=pathlib.Path(__file__).resolve().parent
def read_guard(f):
 try:return json.loads(f.read_text())
 except (json.JSONDecodeError,FileNotFoundError):return {}
rows=[]
for f in sorted((P/'runs').glob('*/*/measurement.json')):
 out=f.parent;gs=[g for g in out.glob('guard*.json') if read_guard(g).get('performance_valid') and read_guard(g).get('phase')=='finished']
 if not gs:continue
 g=gs[-1];guard=json.load(open(g));assert guard.get('workload_exit_code')==0
 m=json.load(open(f));c=json.load(open(out/'correctness.json'));route=json.load(open(out/'candidate_route.json'));i=int(out.parent.name)
 tf=P/('tasks_'+out.name if out.name in ('v2','v3','v4','v5','v6','v7','v8') else 'tasks')/f'{i:02d}.json';t=json.load(open(tf));tim=m['timing'];profiles=tim['retained_profiles'];assert len(profiles)==15 and tim['latency_ms']>0
 names=collections.Counter();aicpu=0
 for pr,ms in zip(profiles,tim['latency_samples_ms'],strict=True):
  ts=pr['profile_kernel_tasks'];ids=[(x['task_id'],x.get('context_id'),tuple(x.get('stream_ids',[]))) for x in ts];assert len(ids)==len(set(ids));assert math.isclose(sum(x['duration_us'] for x in ts)/1000,ms,abs_tol=1e-7)
  for x in ts:
   names[x['name']]+=x['duration_us']/15
   if 'AICPU' in x['core_roles']:aicpu+=x['duration_us']/15
 assert c['status']=='verified' and c['inputs_preserved']
 us=tim['latency_ms']*1000;ratio=us/t['old_public']['tileops_latency_us']
 rows.append({'ordinal':i,'version':out.name,'op':t['key'][1],'case':t['key'][2],'old_baseline_us':t['old_public']['latency_us'],'new_baseline_us':us,'old_ratio':t['old_public']['ratio'],'new_ratio_with_published_tileops':ratio,'proposed_goals':sorted(set(route['targets']+([3] if ratio<=2 else []))),'change':route['change'],'windows':15,'aicpu_us':aicpu,'tasks_per_window':sum(len(pr['profile_kernel_tasks']) for pr in profiles)/15,'measurement_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'guard_sha256':hashlib.sha256(g.read_bytes()).hexdigest(),'top_tasks':names.most_common(4),'out':str(out)})
print(json.dumps({'verified_measurements':len(rows),'unique_cases':len({r['ordinal'] for r in rows}),'rows':rows},indent=2))
