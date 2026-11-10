import pathlib,json,datetime,sqlite3,os,subprocess,hashlib
P=pathlib.Path(__file__).resolve().parent;D=P.parent
now=datetime.datetime.now(datetime.timezone.utc);s=json.loads((D/'STATE.json').read_text());live=json.loads((D/'LIVE.json').read_text());prev=json.loads((P/'DISPATCH_SUMMARY.json').read_text())
deadline=datetime.datetime.fromisoformat(s['deadline_utc']);accepted=len(json.loads((D/'ACCEPTED.json').read_text())['rows']);hours=(deadline-now).total_seconds()/3600
prev.update(at=now.isoformat(),accepted=accepted,remaining=700-accepted,required_rate=(700-accepted)/hours if hours>0 else None,deadline_utc=s['deadline_utc'],observed_accepted_last_hour=live.get('accepted_last_hour'),execution_workers=live.get('execution_workers'),execution_alive=live.get('execution_workers_alive'),coordinators=live.get('coordinator_workers'),monitor_snapshot_at=live.get('at'),target_execution_workers=24,proportional_expansion_applicable=False)
prev.update(deadline_kind='delivery_sla',sla_missed=hours<0,overdue_seconds=max(0,-hours*3600),dispatch_stop_at=s.get('dispatch_stop_at'),session_end_stops_dispatch=False)
prev['fleet']=json.loads((D/'FLEET_REFILL_LIVE.json').read_text())
r=subprocess.run(['python3',str(P/'capture_replacements.py')],capture_output=True,text=True)
if r.returncode==0:prev['native_replacement_summary']=json.loads(r.stdout)
else:prev['native_replacement_capture_error']=r.stderr[-500:]
if (P/'LATEST_ACTION.json').exists():prev['last_substantive_action']=json.loads((P/'LATEST_ACTION.json').read_text())
if (D/'NOTIFICATION_ROUTER_LIVE.json').exists():
 routing=json.loads((D/'NOTIFICATION_ROUTER_LIVE.json').read_text());prev['notification_health']={'at':routing.get('at'),'controller_notifications':routing.get('controller_notifications'),'pending':routing.get('waiting_for_delegated_root'),'acceptance_receiver_active':bool(routing.get('targets',{}).get('acceptance_root'))}
for n in ['DISPATCH_SUMMARY.json','HEARTBEAT.json']:
 target=P/n;tmp=P/(n+'.tmp.'+str(os.getpid()));tmp.write_text(json.dumps(prev,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,target)
with (P/'HEARTBEAT_HISTORY.jsonl').open('a') as f:f.write(json.dumps(prev,ensure_ascii=False)+'\n')
print(json.dumps({k:prev[k] for k in ['at','accepted','remaining','required_rate','observed_accepted_last_hour','execution_workers','execution_alive','coordinators']}))

# Controller requests are explicit authorized directives, not worker results.
for request,ack in [("CONTROLLER_DIRECTIVE_20261027T0500.txt","CONTROLLER_ACTION_ACK_0500.json"),("AUTONOMY_REPAIR_REQUEST_0502.txt","AUTONOMY_REPAIR_ACK_0502.json")]:
 if (D/request).exists():
  digest=hashlib.sha256((D/request).read_bytes()).hexdigest()
  receipt=json.loads((D/ack).read_text()) if (D/ack).exists() else {}
  if receipt.get('request_sha256')!=digest:
   print('PENDING_CONTROLLER_DIRECTIVE',request,'sha256='+digest,'ACK missing or stale; agent must read current directive')
