import json,pathlib,time,argparse,datetime
P=pathlib.Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');a=ap.parse_args()
def read(p):
 try:return json.loads(p.read_text())
 except FileNotFoundError:return {}
while True:
 total=0
 print(datetime.datetime.now(datetime.timezone.utc).isoformat())
 for h in ['server1','server2']:
  plan=read(P/h/'PLAN.json');d={}
  for f in (P/h/'runs').glob('*/a*/ARCHIVE.json'):
   r=read(f);tid=f.parent.parent.name
   if r.get('accepted') or tid not in d or (not d[tid].get('accepted') and r.get('at',0)>d[tid].get('at',0)):d[tid]=r
  s=read(P/h/'REMOTE_STATUS.json');n=sum(bool(x.get('accepted')) for x in d.values());total+=n
  print(h, str(n)+'/'+str(len(plan.get('rows',[]))), 'accepted;', sum(not x.get('accepted',False) for x in d.values()),'pending;',max(0,s.get('completed',0)-n),'awaiting archive;',s.get('status','not launched'),'free GiB',round(s.get('free_GiB',0),2),'heartbeat age',round(time.time()-s.get('received_at',s['at'])) if s.get('at') else 'unknown')
 print('TOTAL',str(total)+'/1135', '['+'#'*int(total/1135*40)+'-'*(40-int(total/1135*40))+']',flush=True)
 if not a.watch:break
 time.sleep(10)
