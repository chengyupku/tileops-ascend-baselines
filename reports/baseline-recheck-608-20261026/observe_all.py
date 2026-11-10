import pathlib,json,sqlite3,time,datetime,collections
from observe import read,alive
R=pathlib.Path(__file__).resolve().parent

def once():
 now=time.time();plan=read(R/'PLAN.json');start=datetime.datetime.fromisoformat(plan['started_at']).timestamp();jobs={g['native']['id']:(g['name'],pathlib.Path(g['directory'])) for g in read(R/'JOBS.json')['jobs']}
 jobs.update({read(p)['id']:(p.parent.name,p.parent) for p in R.glob('audit_*/JOB.json')});db=sqlite3.connect('file:/root/dyq/codex-workers/workers.sqlite?mode=ro',uri=True);allj={i:json.loads(d) for i,d in db.execute('select id,data from jobs')};db.close();workers=[]
 for jid,(name,p) in jobs.items():
  j=allj[jid];e=pathlib.Path(j['directory'])/'events.jsonl';commands=[];errors=[];lastmsg=''
  for l in e.read_text(errors='replace').splitlines() if e.exists() else []:
   try:v=json.loads(l)
   except ValueError:continue
   it=v.get('item',{})
   if v.get('type')=='item.completed' and it.get('type')=='command_execution':commands.append(it)
   if v.get('type') in ('error','turn.failed'):errors.append(v)
   if it.get('type')=='agent_message':lastmsg=it.get('text','')[-500:]
  scope=read(p/'SCOPE.json',{'rows':[]})['rows'];out=read(p/'AUDIT.json',{}).get('rows',[]) if name.startswith('audit') else [l for l in (p/'decisions.jsonl').read_text().splitlines() if l.strip()]
  workers.append(dict(name=name,id=jid,status=j['status'],effort=j['effort'],alive=alive(j),created_at=j['created_at'],finished_at=j.get('finished_at'),wall_s=round((j.get('finished_at') or now)-j['created_at'],1),assigned=len(scope),submitted=len(out),commands=len(commands),errors=errors,failed_commands=sum(x.get('exit_code',0)!=0 for x in commands),event_age_s=round(now-e.stat().st_mtime,1) if e.exists() else None,heartbeat=read(p/'HEARTBEAT.json'),last_message=lastmsg))
 act=[w for w in workers if w['status'] in ('starting','running','stopping')];accepted=read(R/'PARENT_ACCEPTED.json',[]);elapsed=now-start
 s=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=round(elapsed,1),accepted=len(accepted),remaining=608-len(accepted),accepted_rate_per_hour=round(len(accepted)*3600/elapsed,1),active=len(act),live=sum(w['alive'] for w in act),active_survival=sum(w['alive'] for w in act)/len(act) if act else None,completed=sum(w['status']=='completed' for w in workers),failed=sum(w['status'] in ('failed','lost','timed_out') for w in workers),jobs=len(workers),workers=workers)
 (R/'ALL_LIVE.json').write_text(json.dumps(s,ensure_ascii=False,indent=2));summary={k:v for k,v in s.items() if k!='workers'}
 with (R/'ALL_HISTORY.jsonl').open('a') as f:f.write(json.dumps(summary,ensure_ascii=False)+'\n')
 print(json.dumps(summary,ensure_ascii=False),flush=True);return s
if __name__=='__main__':
 while True:
  s=once()
  if not s['active']:break
  time.sleep(20)
