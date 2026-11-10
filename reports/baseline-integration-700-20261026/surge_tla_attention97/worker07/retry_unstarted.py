from resume_attempt import run,status,HOSTS,P,now
from collect_resume import collect
import json,concurrent.futures,datetime,time
keys=json.loads((P/'REMAINING.json').read_text());deadline=datetime.datetime(2026,10,27,5,20,tzinfo=datetime.timezone.utc)
# Rotate no-child requests across compatible hosts. Never retry a device execution
# without a separately established implementation/deployment condition change.
for turn in range(4):
 progress=False
 for key in keys:
  if datetime.datetime.now(datetime.timezone.utc)>=deadline:break
  summary=collect();own=[a for a in summary['attempts'] if a['key']==key]
  if any(a['child'] for a in own):continue
  with concurrent.futures.ThreadPoolExecutor(3) as e:states=list(e.map(status,HOSTS))
  def score(s):
   try:d=json.loads(s['stdout'])
   except:return (9,9,9)
   if d.get('state') not in ['idle','busy']:return (8,8,8)
   count=sum(a['host']==s['host'] for a in own)
   return (0 if d.get('state')=='idle' else 1,count,HOSTS.index(s['host']))
  chosen=min(states,key=score);host=chosen['host']
  if score(chosen)[0]>=8:continue
  (P/'retry_host_selection.json').write_text(json.dumps(dict(utc=now(),key=key,states=states,selected=host),indent=2))
  run(key[1],key[2],host,'No prior device child for exact key. Rotating compatible host/queue after no-child expiry; fresh occupancy snapshots retained. Prior attempts: '+str([(a['host'],a['guard_exit']) for a in own]))
  collect();progress=True
 if not progress:break
print('UNSTARTED_RETRY_PASS_DONE',flush=True)
