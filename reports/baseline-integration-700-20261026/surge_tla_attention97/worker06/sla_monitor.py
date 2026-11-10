from pathlib import Path
import json,datetime,time
P=Path(__file__).resolve().parent
O=P/'local_runs/llama-8b-serving-step-float16-float16-sla-20261027T0528-unstarted'
D=datetime.datetime(2026,10,27,5,31,tzinfo=datetime.timezone.utc)
while True:
 now=datetime.datetime.now(datetime.timezone.utc)
 done=(O/'RESULT.json').exists()
 state={'at':now.isoformat(),'key':['tilelang-ascend','GQAPagedFwdOp','llama-8b-serving-step-float16-float16'],'phase':'case_finished' if done else 'original_guard_execution','last_output':str(O),'blocker':None if done else 'await original checker within 180-second case budget','next':'merge exact receipt into full-scope RESULT','sla_deadline':D.isoformat(),'sla_overdue':now>D,'accepted_added':0}
 (P/'HEARTBEAT.json').write_text(json.dumps(state,indent=2))
 if now>=D and not (P/'SLA_0531_SNAPSHOT.json').exists():
  state['guard_running']=json.loads((O/'GUARD_RUNNING.json').read_text()) if (O/'GUARD_RUNNING.json').exists() else None
  state['guard_result']=json.loads((O/'guard.json').read_text()) if (O/'guard.json').exists() else None
  state['checker_exists']=(O/'checker.json').exists()
  (P/'SLA_0531_SNAPSHOT.json').write_text(json.dumps(state,indent=2))
 if done and now>=D:break
 time.sleep(min(20,max(0.1,(D-now).total_seconds())) if now<D else 5)
print(json.dumps(state))
