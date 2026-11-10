from resume_attempt import run,status,HOSTS,P,now
from collect_resume import collect
import json,concurrent.futures,datetime
keys=json.loads((P/'REMAINING.json').read_text())
# Exact unexecuted attention cases first, then original cosine cases. No automatic
# replay after any device child has produced a checker or deployment exception.
keys=sorted(keys,key=lambda k:(k[1]!='GQAVarlenFwdOp', 'short' not in k[2], k[2]))
for key in keys:
 if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,10,27,5,20,tzinfo=datetime.timezone.utc):break
 summary=collect();done=any(a['key']==key and a['child'] for a in summary['attempts'])
 if done:continue
 with concurrent.futures.ThreadPoolExecutor(3) as e:states=list(e.map(status,HOSTS))
 def score(s):
  try:d=json.loads(s['stdout'])
  except:return 9
  return 0 if d.get('state')=='idle' else (1 if s['host']=='local' and d.get('state')=='busy' else 2 if d.get('state')=='busy' else 8)
 host=min(states,key=score)['host'];(P/'batch_last_host_selection.json').write_text(json.dumps(dict(utc=now(),key=key,states=states,selected=host),indent=2))
 run(key[1],key[2],host,'First device execution sought for this exact key; fresh three-host compatibility/status, original default target retained. Any prior attempts were no-child only.')
 collect()
print('BATCH_PASS_DONE',flush=True)
