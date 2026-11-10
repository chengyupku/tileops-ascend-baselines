"""Foreground CPU-only dispatch heartbeat; existing fleet owns native launches."""
import pathlib,datetime,time,json,subprocess,sqlite3,os
P=pathlib.Path(__file__).resolve().parent;D=P.parent;cycle=0
while True:
 now=datetime.datetime.now(datetime.timezone.utc);state=json.loads((D/'STATE.json').read_text());deadline=datetime.datetime.fromisoformat(state['deadline_utc'])
 if not state.get('enabled') or state.get('phase') in ('paused','stopped','complete') or now>=deadline:
  print(json.dumps({'at':now.isoformat(),'watch_stopped':'state_or_original_deadline'},ensure_ascii=False),flush=True);break
 subprocess.run(['python3',str(P/'pulse.py')],check=True)
 if cycle%5==0:
  r=subprocess.run(['python3',str(D/'refill_fleet.py')],capture_output=True,text=True)
  service=subprocess.run(['systemctl','show','baseline700-surge-refill.service','baseline700-20261026-sample.service','-p','Id','-p','Result','-p','ExecMainStatus','-p','ExecMainExitTimestamp'],capture_output=True,text=True)
  row={'at':now.isoformat(),'action':'safe existing refill and coordinator restoration check','refill_rc':r.returncode,'refill':r.stdout[-3500:],'error':r.stderr[-1000:],'service':service.stdout,'scope_guard':'No new scope or changed condition manufactured; inactive exhausted groups remain disabled'}
  with (P/'WATCH_ACTIONS.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
  print(json.dumps({'refill_rc':r.returncode,'cycle':cycle,'action':'existing refill supervision check'},ensure_ascii=False),flush=True)
 cycle+=1
 time.sleep(min(60,max(0,(deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds())))
