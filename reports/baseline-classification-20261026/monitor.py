"""Read-only native lifecycle observation and binary-output aggregation for one hour."""
import pathlib,json,sqlite3,time,datetime,collections,csv,os
P=pathlib.Path(__file__).resolve().parent
def read(p,default=None):
 try:return json.load(open(p))
 except (OSError,ValueError):return default
def write(p,v):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2));tmp.replace(p)
def alive(j):
 try:
  s=pathlib.Path('/proc/'+str(j['child_pid'])+'/stat').read_text().rsplit(')',1)[1].split()
  return s[0]!='Z' and pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()+':'+s[19]==j['child_stamp']
 except Exception:return False
scopes={str(p.parent):{tuple(r['key']) for r in read(p)['rows']} for p in P.glob('group*/batch*/SCOPE.json')}
expected=set().union(*scopes.values());first=read(P/'FIRST_OBSERVED.json',{});anchor=read(P/'MONITOR_ANCHOR.json');previous_print=0
while True:
 now=time.time();db=sqlite3.connect('file:/root/dyq/codex-workers/workers.sqlite?mode=ro',uri=True)
 jobs=[];coords=[]
 for ident,status,raw in db.execute('SELECT id,status,data FROM jobs'):
  j=json.loads(raw);ws=j.get('write',[])
  if any(p in scopes for p in ws):jobs.append(j)
  elif any(p==str(P/f'group{i}') for i in range(1,5) for p in ws):coords.append(j)
 db.close()
 if not anchor and any(alive(j) for j in jobs):
  at=min(j['created_at'] for j in jobs if alive(j));anchor={'start_epoch':at,'end_epoch':at+3600,'start_utc':datetime.datetime.fromtimestamp(at,datetime.timezone.utc).isoformat(),'basis':'first genuinely running classifier native creation; no reused decisions credited'};write(P/'MONITOR_ANCHOR.json',anchor)
 valid={};invalid=[]
 for directory,keys in scopes.items():
  f=pathlib.Path(directory)/'decisions.jsonl'
  if not f.exists():continue
  lines=f.read_text().splitlines();seen=set()
  for i,line in enumerate(lines):
   try:r=json.loads(line);k=tuple(r['key'])
   except (ValueError,KeyError,TypeError):
    if i<len(lines)-1:invalid.append({'path':str(f),'line':i+1,'reason':'invalid_json'})
    continue
   if k not in keys or k in seen or r.get('decision') not in ('要接','不要接') or not r.get('reason') or not r.get('evidence'):
    invalid.append({'path':str(f),'line':i+1,'key':list(k),'reason':'scope_duplicate_or_missing_binary_evidence'});continue
   seen.add(k);valid[k]=dict(r,worker_output=str(f));ident=json.dumps(k,ensure_ascii=False)
   first.setdefault(ident,now)
 write(P/'FIRST_OBSERVED.json',first)
 rows=[]
 for j in jobs:
  ep=pathlib.Path(j['directory'])/'events.jsonl';errors=[];commands=0;last_cmd=None;last_msg=None
  if ep.exists():
   for line in ep.read_text(errors='replace').splitlines():
    try:e=json.loads(line)
    except ValueError:continue
    if e.get('type') in ('error','turn.failed'):errors.append(e)
    it=e.get('item',{})
    if e.get('type')=='item.completed' and it.get('type')=='command_execution':commands+=1;last_cmd=it.get('command','')[:1200]
    if it.get('type')=='agent_message':last_msg=it.get('text','')[:1200]
  ws=next(p for p in j['write'] if p in scopes);nj=sum(k in valid for k in scopes[ws]);live=alive(j)
  rows.append({'id':j['id'],'status':j['status'],'alive':live,'home':j['home']['name'],'model':j.get('model'),'effort':j.get('effort'),'directory':ws,'created_at':j['created_at'],'finished_at':j.get('finished_at'),'age_s':now-j['created_at'],'last_event_age_s':now-ep.stat().st_mtime if ep.exists() else None,'commands':commands,'last_command':last_cmd,'last_message':last_msg,'errors':errors[-2:],'classified':nj,'assigned':len(scopes[ws])})
 claims=[r for r in rows if r['status'] in ('starting','running','stopping','cancelling')];live=sum(r['alive'] for r in rows);elapsed=now-anchor['start_epoch'] if anchor else 0
 windows={str(sec):sum(now-sec<=first.get(json.dumps(k,ensure_ascii=False),0)<=now for k in valid) for sec in (60,300,900,3600)}
 alerts=[]
 if any(r['effort']!='medium' for r in rows):alerts.append('effort_not_medium')
 if any(not r['alive'] and r['age_s']>30 for r in claims):alerts.append('native_active_pid_absent')
 failures=[r for r in rows if r['status'] in ('failed','timed_out','lost')]
 if failures:alerts.append('terminal_failures_require_common_cause_analysis')
 if invalid:alerts.append('invalid_output_rows')
 if elapsed>600 and len(valid)<500*elapsed/3600 and len(valid)<len(expected):alerts.append('below500_cumulative_rate')
 snapshot={'at':now,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'anchor':anchor,'elapsed_s':elapsed,'monitor_complete':bool(anchor and now>=anchor['end_epoch']),'target_per_hour':500,'expected_new_pairs':len(expected),'new_classified':len(valid),'remaining':len(expected)-len(valid),'decision_counts':dict(collections.Counter(r['decision'] for r in valid.values())),'confidence_counts':dict(collections.Counter(r.get('confidence') for r in valid.values())),'cumulative_rate_per_hour':len(valid)*3600/elapsed if elapsed else 0,'window_counts':windows,'workers_spawned':len(rows),'worker_statuses':dict(collections.Counter(r['status'] for r in rows)),'actual_live_workers':live,'live_claims':len(claims),'live_claim_survival':sum(r['alive'] for r in claims)/len(claims) if claims else None,'coordinator_statuses':dict(collections.Counter(j['status'] for j in coords)),'workers':rows,'invalid':invalid,'alerts':alerts}
 write(P/'LIVE.json',snapshot)
 with (P/'HISTORY.jsonl').open('a') as f:f.write(json.dumps({k:v for k,v in snapshot.items() if k not in ('workers','invalid')},ensure_ascii=False)+'\n')
 with (P/'NEW_DECISIONS.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['baseline','operator','case_id','decision','reason','confidence','evidence','first_observed_utc'])
  for k,r in sorted(valid.items()):w.writerow([*k,r['decision'],r['reason'],r.get('confidence'),json.dumps(r['evidence'],ensure_ascii=False),datetime.datetime.fromtimestamp(first[json.dumps(k,ensure_ascii=False)],datetime.timezone.utc).isoformat()])
 if now-previous_print>=45 or snapshot['monitor_complete']:
  print(json.dumps({k:snapshot[k] for k in ('utc','elapsed_s','new_classified','remaining','actual_live_workers','worker_statuses','cumulative_rate_per_hour','alerts','monitor_complete')},ensure_ascii=False),flush=True);previous_print=now
 if snapshot['monitor_complete']:write(P/'ONE_HOUR_RESULT.json',snapshot);break
 time.sleep(15)
