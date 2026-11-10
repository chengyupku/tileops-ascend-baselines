import json,sys,time,datetime,collections
from pathlib import Path
host=sys.argv[1];R=Path('reports/baseline-cache-20261027');H=R/host;D=R/'status-20261029T012849'
now=time.time();offset=0 if host=='local' else 93.08
start=datetime.datetime(2026,10,28,20,0,tzinfo=datetime.timezone.utc).timestamp()
rows=[json.loads(x) for x in (H/'journal.jsonl').read_text().splitlines() if x.strip()]
recent=[x for x in rows if start<=x.get('at',0)+offset<=now+offset]
attempts=[x for x in recent if x.get('wall_seconds') is not None and x['status'] not in ('resource_wait','temporarily_skipped_library')]
intervals=sorted((max(start,x['at']+offset-x['wall_seconds']),min(now+offset,x['at']+offset)) for x in attempts)
merged=[]
for a,b in intervals:
 if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
 else:merged.append([a,b])
gaps=[];last=start
for a,b in merged:
 if a-last>120:gaps.append({'from':datetime.datetime.fromtimestamp(last,datetime.timezone.utc).isoformat(),'to':datetime.datetime.fromtimestamp(a,datetime.timezone.utc).isoformat(),'minutes':(a-last)/60})
 last=max(last,b)
l=json.loads((H/'LIVE.json').read_text());active=l.get('phase')=='running'
if now+offset-last>120:gaps.append({'from':datetime.datetime.fromtimestamp(last,datetime.timezone.utc).isoformat(),'to':datetime.datetime.fromtimestamp(now+offset,datetime.timezone.utc).isoformat(),'minutes':(now+offset-last)/60,'may_include_current_attempt':active})
by={}
for x in attempts:
 if x['status']=='cached':continue
 key=(x['key'][0],x['status'],str(x.get('error') or '')[:260])
 v=by.setdefault(key,{'count':0,'seconds':0,'examples':[]});v['count']+=1;v['seconds']+=x['wall_seconds']
 if len(v['examples'])<2:v['examples'].append({'key':x['key'],'id':x['id'],'error':x.get('error')})
out={'host':host,'window_start_utc':datetime.datetime.fromtimestamp(start,datetime.timezone.utc).isoformat(),'checked_utc':datetime.datetime.fromtimestamp(now+offset,datetime.timezone.utc).isoformat(),'clock_offset_correction_seconds':offset,'physical_journal_status':dict(collections.Counter(x['status'] for x in recent)),'cache_growth':len({x['id'] for x in recent if x['status']=='cached'}),'attempts':len(attempts),'finished_attempt_union_seconds':sum(b-a for a,b in merged),'window_seconds':now+offset-start,'queue_wait_seconds':sum(x.get('wall_seconds',0) for x in recent if x['status']=='resource_wait'),'gaps_between_attempts_over2min':gaps,'top_failed_attempt_wall_groups':[dict(library=k[0],status=k[1],error=k[2],**v) for k,v in sorted(by.items(),key=lambda kv:-kv[1]['seconds'])[:10]],'current_live':l,'last_cached_at':max((x['at'] for x in rows if x['status']=='cached'),default=None)}
(D/(host+'-overnight.json')).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['current_live','top_failed_attempt_wall_groups']},ensure_ascii=False))
