import collections,json,pathlib,sys,time
h=sys.argv[1];R=pathlib.Path('reports/baseline-cache-20261027');H=R/h;D=R/'inspection-20261029T001544'
read=lambda p:json.loads(p.read_text())
rows=[json.loads(s) for s in (H/'journal.jsonl').read_text().splitlines() if s.strip()]
plan=read(R/'PLAN.json');owned={t['id'] for t in plan['pairs'] if t['host']==h}
launch=read(D/(h+'-LAUNCH.json'));after=[r for r in rows if r['id'] in owned and r.get('at',0)>=launch['at']]
incoming={t['id'] for t in read(D/'ASSIGNMENT_CHANGES.json')['changes'] if t['to']==h}
old=read(R/'inspection-20261028T201544'/(h+'-FP16_HOLDS.json'));held={t['id'] for t in old['added']}
oldrows=[r for r in rows if r['id'] in held and r.get('at',0)>=old['at']]
result={'at':time.time(),'host':h,'launch_at':launch['at'],'post_launch_counts':dict(collections.Counter(r['status'] for r in after)),
    'post_launch_events':[{k:r.get(k) for k in ['id','key','at','status','wall_seconds','error','measurement','guard']} for r in after],
    'incoming_outcomes':{r['id']:r['status'] for r in after if r['id'] in incoming},
    'previous_111_hold_count':len(held),'previous_111_hold_consumption':dict(collections.Counter(r['status'] for r in oldrows)),
    'previous_111_held_device_attempts':[r for r in oldrows if r.get('wall_seconds') is not None]}
if h=='local':
    old=read(R/'inspection-20261028T161544/ADMISSION_HOLDS.json');held={r['id'] if isinstance(r,dict) else r for r in old['added']}
    oldrows=[r for r in rows if r['id'] in held and r.get('at',0)>=old['at']]
    result['previous_356_hold_consumption']=dict(collections.Counter(r['status'] for r in oldrows));result['previous_356_held_device_attempts']=[r for r in oldrows if r.get('wall_seconds') is not None]
(D/(h+'-EFFECT.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
