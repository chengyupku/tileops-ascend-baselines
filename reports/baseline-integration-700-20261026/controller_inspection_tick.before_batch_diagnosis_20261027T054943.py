#!/usr/bin/env python3
"""Wake the explicitly authorized throughput controller every five minutes."""
import datetime,fcntl,json,os,pathlib,subprocess,sys,time
D=pathlib.Path(__file__).resolve().parent

def read(p,default):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return default

def put(p,x):
 tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()));tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)

def prepare(state,snapshot,config,now):
 if not config.get('enabled') or not state.get('enabled') or state.get('phase') in ('paused','stopped','complete'):return None
 if snapshot.get('parent_accepted_completed',0)>=700:return None
 at=datetime.datetime.fromtimestamp(now,datetime.timezone.utc).isoformat()
 stamp=snapshot.get('at');age=None
 if stamp:
  try:age=now-datetime.datetime.fromisoformat(stamp).timestamp()
  except ValueError:pass
 fields=['at','parent_accepted_completed','execution_workers','execution_workers_alive','coordinator_workers','accepted_last_hour','required_accepted_per_hour','flags']
 summary={k:snapshot.get(k) for k in fields};summary['snapshot_age_seconds']=age;summary['snapshot_stale']=age is None or age>120
 return {'inspection_id':at,'summary':summary}

def main():
 with (D/'CONTROLLER_INSPECTION.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);state=read(D/'STATE.json',{});config=read(D/'CONTROLLER_INSPECTION_CONFIG.json',{});record=prepare(state,read(D/'LIVE.json',{}),config,time.time())
  if record is None:print(json.dumps({'skipped':'paused, disabled or accepted scope complete'}));return
  previous=read(D/'LAST_CONTROLLER_INSPECTION.json',{});record['previous_controller_inspection_at']=previous.get('at')
  message=('[baseline700 controller five-minute inspection] User explicitly requires THIS throughput-controller conversation to inspect at least every five minutes. This is aggregate oversight, not individual-result acceptance. Inspection '+record['inspection_id']+'. Read the current task memory header/owned controller_autonomy_inspection projection and baseline operating rules as needed; do not dump giant historical STATE files. Read '+str(D/'LIVE.json')+', FLEET_REFILL_LIVE.json, active role/READY/PUBLISH queue counts and service success/freshness. Verify native PID/start identity. Measure accepted throughput, slot occupancy, terminal failures, queue/host phases and significant common waste; inspect agent commands only when throughput is insufficient. Do not review any individual worker result or write ACCEPTED/providers. YOU are the repairer: personally investigate and repair throughput, scheduler, queue, liveness and common agent-waste problems. Do not hand the diagnosis or repair responsibility to an integration root or count a queued request as a repair. Integration roots continue integration, per-item evidence review and acceptance. Before each new action batch, tell the user what you will do, why and what you will change, then execute; this is prior notice, not a requirement to ask permission. Prioritize one or two largest common bottlenecks, personally implement concrete scoped fixes and verify their effect, preserve original baseline/top-level-interface-only/full-checker boundaries and three-host failover. Use existing native script-managed refill; no duplicate owners, no unchanged failures or completed-case reruns, no fake agents to fill24. Original05:31UTC delivery SLA remains missed if incomplete, not reset and not an automatic stop. Respect any later user pause. Worker completion notices remain routed to lower roots. Record this inspection_id, actual checked_at, current aggregate findings, concrete action or evidence why no legal action exists, action owner, follow-up evidence and next check in '+str(D/'LAST_CONTROLLER_INSPECTION.json')+'; append dated entry to CONTROLLER_INSPECTIONS.jsonl. Queue delivery is not an inspection acknowledgement. Snapshot '+json.dumps(record,ensure_ascii=False))
  if '--dry-run' in sys.argv:print(json.dumps({'dry_run':True,'target_thread':config['thread'],'target_home':config['home'],'snapshot':record},ensure_ascii=False));return
  result=subprocess.run(['/root/dyq/agent-bin/'+config['home'],'queue','--thread',config['thread'],'--message',message],capture_output=True,text=True,timeout=30)
  record.update(target_thread=config['thread'],target_home=config['home'],queue_returncode=result.returncode,queue_receipt=result.stdout.strip(),queue_error=result.stderr[-600:],completed_inspection=False)
  put(D/'CONTROLLER_INSPECTION_TICK.json',record)
  with (D/'CONTROLLER_INSPECTION_TICKS.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
  print(json.dumps(record,ensure_ascii=False))
  if result.returncode:raise SystemExit(result.returncode)
if __name__=='__main__':main()
