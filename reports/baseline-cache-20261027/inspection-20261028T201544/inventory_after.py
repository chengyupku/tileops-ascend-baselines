import json, time, sys, subprocess, statistics, collections, datetime
from pathlib import Path
host=sys.argv[1];assert host in ('local','server1','server2')
sys.path.insert(0,'tools')
from run_baseline_cache import assigned_history, completed_ids, quarantine_key, identity
R=Path('reports/baseline-cache-20261027');H=R/host;D=R/'inspection-20261028T201544';D.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text())
samples=[]
for i in range(2):
    started=time.time()
    p=subprocess.run(['npu-guard','status','--json'],capture_output=True,text=True,timeout=15)
    g=json.loads(p.stdout);l=read(H/'LIVE.json');meta=g.get('reservation',{}).get('metadata',{})
    native={}
    for name,x in [('supervisor',l.get('identity')),('guard',l.get('guard_identity')),('child',meta.get('child'))]:
        if not x:continue
        try:
            a=identity(x['pid']);st=Path('/proc')/str(x['pid'])/'stat';fields=st.read_text().rsplit(')',1)[1].split()
            native[name]={'matches':all(a[k]==v for k,v in x.items() if k in a),'state':fields[0]}
        except OSError:native[name]={'matches':False,'state':'exited'}
    samples.append({'at':started,'guard':g,'live':{k:l.get(k) for k in ['at','phase','key','pair_phase','pair_elapsed_seconds']},'native':native})
    if i<1:time.sleep(max(0,5-(time.time()-started)))
now=time.time();tasks=[t for t in read(R/'PLAN.json')['pairs'] if t['host']==host];owned={t['id'] for t in tasks}
rows=[json.loads(x) for x in (H/'journal.jsonl').read_text().splitlines() if x.strip()]
caches=[json.loads(x) for x in (H/'CACHE.jsonl').read_text().splitlines() if x.strip()]
latest,cached=assigned_history(tasks,rows,caches);q=read(H/'QUARANTINE.json');done=completed_ids(latest,read(H/'RETRY.json'),q,cached)
ready=[t for t in tasks if t['id'] not in done and not quarantine_key(t,q)];held=[t for t in tasks if t['id'] not in cached and quarantine_key(t,q)]
windows={}
for hours in [.5,1,4]:
    recent=[x for x in rows if x['id'] in owned and x.get('at',0)>=now-hours*3600]
    attempts=[x for x in recent if x.get('wall_seconds') is not None and x['status'] not in ('resource_wait','temporarily_skipped_library')]
    status=collections.Counter(x['status'] for x in recent)
    cached_n=len({x['id'] for x in recent if x['status']=='cached'});tested=len({x['id'] for x in attempts})
    windows[str(hours)]={'cached':cached_n,'cached_per_hour':cached_n/hours,'tested':tested,'tested_per_hour':tested/hours,'status':dict(status),'mean_attempt_seconds':statistics.mean(x['wall_seconds'] for x in attempts) if attempts else None,'queue_wall_seconds':sum(x.get('wall_seconds',0) for x in recent if x['status']=='resource_wait'),'eta_queue_hours':len(ready)/(tested/hours) if tested else None,'library_attempts':dict(collections.Counter(x['key'][0] for x in attempts))}
util={}
for metric in ['npu','aicore','cube','vector','hbm_bw']:
    values=[s['guard']['utilization'][metric]['value'] for s in samples if s['guard'].get('utilization',{}).get(metric,{}).get('rc')==0]
    util[metric]={'n':len(values),'mean':statistics.mean(values) if values else None,'max':max(values) if values else None,'zero_samples':values.count(0),'nonzero_samples':sum(x>0 for x in values)}
summary={'at':now,'host':host,'host_time':datetime.datetime.fromtimestamp(now,datetime.timezone.utc).isoformat(),'sample_seconds':samples[-1]['at']-samples[0]['at'],'utilization':util,'reservation_locked_samples':sum(s['guard'].get('reservation',{}).get('locked') is True for s in samples),'sample_count':len(samples),'sample_phases':dict(collections.Counter(s['live'].get('pair_phase',{}).get('phase','none') if isinstance(s['live'].get('pair_phase'),dict) else 'none' for s in samples)),'assigned':len(tasks),'cached':len(cached),'ready':len(ready),'held':len(held),'tested_noncache_unheld':len(tasks)-len(cached)-len(ready)-len(held),'ready_libraries':dict(collections.Counter(t['key'][0] for t in ready)),'windows':windows,'current':samples[-1]['live'],'identities':samples[-1]['native']}
import shutil, hashlib
summary['disk']=dict(zip(['total','used','free'],shutil.disk_usage('.')))
summary['plan_sha256']=hashlib.sha256((R/'PLAN.json').read_bytes()).hexdigest()
summary['live_age_seconds']=now-read(H/'LIVE.json')['at']
summary['last_events']=[{k:x.get(k) for k in ('id','key','status','at','wall_seconds','error','measurement','guard')} for x in rows if x['id'] in owned][-12:]
summary['recent_errors']=dict(collections.Counter(str(x.get('error'))[:350] for x in rows if x['id'] in owned and x.get('at',0)>=now-4*3600 and x.get('error')))
summary['ready_by_operator']=dict(collections.Counter('::'.join(t['key'][:2]) for t in ready))
summary['quarantine']=q
(D/(host+'-after.json')).write_text(json.dumps({'summary':summary,'samples':samples},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['last_events','quarantine','ready_by_operator']},ensure_ascii=False))
