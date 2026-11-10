import json,statistics,datetime,collections
from pathlib import Path
h=Path(__file__).resolve().parent;d=json.loads((h/'OVERLAP_TIMELINE_RAW.json').read_text())
def ts(v):return datetime.datetime.fromisoformat(v).timestamp() if v else None
def iso(t):return datetime.datetime.fromtimestamp(t,datetime.timezone.utc).isoformat()
intervals={'prepare':[],'device':[]};byjob={};single=0
for b in d['batches']:
 starts={}
 single+=int(len([e for e in b['events'] if e['event']=='phase_start' and e.get('phase')=='device'])==1)
 for e in b['events']:
  if e['event']=='phase_start':starts[e['id'],e['phase']]=e
  if e['event']=='phase_end' and (e['id'],e['phase']) in starts:
   a=starts[e['id'],e['phase']];v=(a['wall'],e['wall'],b['batch'],e['id'],e.get('returncode'));intervals[e['phase']].append(v);byjob[b['batch'],e['id'],e['phase']]=v
prep=intervals['prepare'];dev=sorted(intervals['device']);overlap=sum(max(0,min(p[1],v[1])-max(p[0],v[0])) for p in prep for v in dev)
gaps=[{'from':iso(a[1]),'to':iso(b[0]),'seconds':b[0]-a[1],'cross_batch':a[2]!=b[2]} for a,b in zip(dev,dev[1:]) if b[0]>a[1]]
records=[]
for a in d['attempts']:
 k=(a['batch'],a['id'],'device');v=byjob.get(k);c=a['correctness'];m=a['measurement'];g=a['guard']
 if not v:continue
 r={'id':a['id'],'key':a['key'],'device_phase_seconds':v[1]-v[0],'exit':v[4],'checker_seconds':c.get('wall_seconds'),'expected_seconds':a['expected_seconds'],'status':m.get('status'),'guard_wait_seconds':ts(g.get('acquired_at'))-ts(g.get('requested_at')) if g.get('acquired_at') and g.get('requested_at') else None}
 if m.get('finished_at') and c.get('finished_at'):r['post_checker_seconds']=m['finished_at']-ts(c['finished_at'])
 if m.get('finished_at') and m.get('started_at'):r['measurement_seconds']=m['finished_at']-m['started_at']
 records.append(r)
valid=[r for r in records if r['status']=='cached'];groups={}
for lib in sorted(set(r['key'][0] for r in valid)):
 x=[r for r in valid if r['key'][0]==lib];groups[lib]={'count':len(x),'device_seconds':sum(r['device_phase_seconds'] for r in x),'checker_seconds':sum(r.get('checker_seconds') or 0 for r in x),'post_checker_seconds':sum(r.get('post_checker_seconds') or 0 for r in x),'expected_seconds':sum(r.get('expected_seconds') or 0 for r in x)}
result={'observed_at':iso(d['at']),'batches':len(d['batches']),'batches_with_one_started_device':single,'prepare_count':len(prep),'prepare_sum_seconds':sum(p[1]-p[0] for p in prep),'prepare_median_seconds':statistics.median(p[1]-p[0] for p in prep),'prepare_overlapped_seconds':overlap,'completed_device_phase_count':len(dev),'device_phase_sum_seconds':sum(p[1]-p[0] for p in dev),'device_phase_median_seconds':statistics.median(p[1]-p[0] for p in dev),'first_device':iso(dev[0][0]),'last_device_end':iso(max(p[1] for p in dev)),'span_hours':(max(p[1] for p in dev)-dev[0][0])/3600,'gap_sum_seconds':sum(x['seconds'] for x in gaps),'largest_gaps':sorted(gaps,key=lambda x:-x['seconds'])[:12],'guard_wait_sum_seconds':sum(r.get('guard_wait_seconds') or 0 for r in records),'failed_device_phase_seconds':sum(r['device_phase_seconds'] for r in records if r['exit']!=0),'by_library_cached_closed_phases':groups,'slowest_checker':sorted([r for r in valid if r.get('checker_seconds')],key=lambda r:-r['checker_seconds'])[:8],'slowest_post_checker':sorted([r for r in valid if r.get('post_checker_seconds')],key=lambda r:-r['post_checker_seconds'])[:8]}
(h/'OVERLAP_ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n');(h/'OVERLAP_ATTEMPT_TIMINGS.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(result,indent=2))
