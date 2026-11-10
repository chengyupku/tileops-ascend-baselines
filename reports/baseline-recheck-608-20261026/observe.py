import json,pathlib,sqlite3,time,datetime,collections,sys
R=pathlib.Path(__file__).resolve().parent

def read(p,d=None):
 try:return json.load(open(p))
 except (OSError,ValueError):return d

def alive(j):
 try:
  s=pathlib.Path(f"/proc/{j['child_pid']}/stat").read_text().rsplit(')',1)[1].split()
  return s[0]!='Z' and pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()+':'+s[19]==j['child_stamp']
 except (OSError,KeyError):return False

def once():
 now=time.time();plan=read(R/'PLAN.json');start=datetime.datetime.fromisoformat(plan['started_at']).timestamp();jobs=read(R/'JOBS.json')['jobs'];first=read(R/'FIRST_OBSERVED.json',{});db=sqlite3.connect('file:/root/dyq/codex-workers/workers.sqlite?mode=ro',uri=True);allj={i:json.loads(d) for i,d in db.execute('select id,data from jobs')};db.close();workers=[];valid={};errors=[]
 for g in jobs:
  j=allj.get(g.get('native',{}).get('id'),g.get('native',{}));p=pathlib.Path(g['directory']);scope={tuple(x['key']) for x in read(p/'SCOPE.json',{'rows':[]})['rows']};rows={};f=p/'decisions.jsonl'
  if f.exists():
   for n,l in enumerate(f.read_text().splitlines()):
    try:x=json.loads(l);k=tuple(x['key'])
    except (ValueError,KeyError,TypeError):errors.append([g['name'],n+1,'invalid_json']);continue
    if k not in scope or k in rows or x.get('decision') not in ('要接','不要接') or not x.get('reason') or not x.get('evidence'):
     errors.append([g['name'],n+1,'scope_duplicate_or_missing_evidence']);continue
    rows[k]=x;valid[k]=dict(x,worker=g['name']);first.setdefault(json.dumps(k,ensure_ascii=False),now)
  events=pathlib.Path(j.get('directory','/nonexistent'))/'events.jsonl';commands=[];lastmsg='';errs=[]
  if events.exists():
   for l in events.read_text(errors='replace').splitlines():
    try:e=json.loads(l)
    except ValueError:continue
    it=e.get('item',{})
    if e.get('type')=='item.completed' and it.get('type')=='command_execution':commands.append(it.get('command',''))
    if it.get('type')=='agent_message':lastmsg=it.get('text','')[-800:]
    if e.get('type') in ('error','turn.failed'):errs.append(e)
  workers.append({'name':g['name'],'id':j.get('id'),'home':j.get('home',{}).get('name'),'status':j.get('status'),'effort':j.get('effort'),'alive':alive(j),'assigned':len(scope),'output_rows':len(rows),'commands':len(commands),'last_command':commands[-1][-1400:] if commands else None,'last_message':lastmsg,'event_age_s':round(now-events.stat().st_mtime,1) if events.exists() else None,'submitted_at':g['submitted_at'],'created_at':j.get('created_at'),'started_at':j.get('started_at'),'finished_at':j.get('finished_at'),'spawn_return_s':round(g['returned_at']-g['submitted_at'],3),'heartbeat':read(p/'HEARTBEAT.json'),'questions':read(p/'QUESTIONS.json'),'errors':errs[-2:]})
 active=[x for x in workers if x['status'] in ('starting','running','stopping')];live=sum(x['alive'] for x in active);failed=[x for x in workers if x['status'] in ('failed','lost','timed_out')];elapsed=now-start
 snap={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':round(elapsed,1),'deadline':plan['deadline'],'output_rows':len(valid),'remaining':608-len(valid),'counts':dict(collections.Counter(x['decision'] for x in valid.values())),'output_rate_per_hour':round(len(valid)*3600/max(elapsed,1),1),'reviewed_accepted':read(R/'REVIEW.json',{}).get('accepted_count',0),'active':len(active),'live':live,'live_survival':live/len(active) if active else None,'terminal_failures':len(failed),'completed_jobs':sum(x['status']=='completed' for x in workers),'workers':workers,'invalid_rows':errors,'metric_note':'output_rows仅结构合格的提交，不代表主agent已验收；存活率分母仅仍应运行任务，已正常完成不算死亡。spawn耗时不冒充API排队时间。'}
 for name,obj in [('LIVE.json',snap),('FIRST_OBSERVED.json',first),('SUBMITTED_DECISIONS.json',list(valid.values()))]:
  tmp=R/(name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(R/name)
 with (R/'HISTORY.jsonl').open('a') as f:f.write(json.dumps({k:v for k,v in snap.items() if k!='workers'},ensure_ascii=False)+'\n')
 print(json.dumps({k:v for k,v in snap.items() if k not in ('workers','invalid_rows','metric_note')},ensure_ascii=False),flush=True)
 return snap
if __name__=='__main__':once()
