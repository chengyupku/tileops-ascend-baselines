from pathlib import Path
import json,datetime,hashlib,subprocess,shlex
P=Path(__file__).resolve().parent
source=(P/'continuation.py').read_text().split('\nallowed=')[0]
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
source=source.replace('-continuation-20261027T043725-%d','-sla-continuation-'+stamp+'-%d')
exec(compile(source,str(P/'continuation.py'),'exec'))
keys=json.loads((P/'SLA_REMAINING.json').read_text());assert len(keys)==1
row=next(r for r in rows if r['key']==keys[0]);assert not row['numerical_success']
original_heartbeat=heartbeat
sla=datetime.datetime.fromisoformat('2026-10-27T05:31:00+00:00').timestamp()
def heartbeat(key,phase,last,blocker=None):
 original_heartbeat(key,phase,last,blocker)
 if time.time()>=sla and not (P/'SLA_053100_SNAPSHOT.json').exists():
  (P/'SLA_053100_SNAPSHOT.json').write_text(json.dumps(dict(sla_utc='2026-10-27T05:31:00+00:00',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),key=key,phase=phase,completed=False,accepted_added=0,continued_after_sla=True,last_output=last),indent=2))
contract=dict(task_type='actual_integration_and_supplemental_verification',exact_keys=keys,owner='baseline700-20261026/surge_tla_attention97/worker02',private_path=str(P),reuse=['RECIPE.md','RECIPE_CONTINUATION.md','CONTINUATION_CPU_CHECKS.json','REPLACEMENT_CLEARANCE.json'],remaining=keys,changed_condition='Exact key previously unstarted; authorized SLA continuation now permits first device attempt.',minimum_acceptance=['complete unchanged original checker','preserved inputs and seed','original source/compiler/library/native identity','guard exit zero'],original_sla='2026-10-27T05:31:00+00:00',accepted_added=0)
(P/'SLA_CONTRACT.json').write_text(json.dumps(contract,indent=2))
# CPU-only delta checks of deployed entry, binding and checker; reuse SDK receipts.
h='openlibing-DevEnv-055409'
files=['run_case.py','launch_continuation.py','adapter.py','capture.py','frozen/TileOPs/scripts/ascend_pair_check.py']
code="from pathlib import Path;import hashlib,json;p=Path("+repr(REMOTE)+");print(json.dumps({n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in "+repr(files)+"}))"
r=subprocess.run([SSH,h,'python3 -c '+shlex.quote(code)],capture_output=True,text=True,check=True)
delta=json.loads(r.stdout);local={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in files}
(P/'SLA_CPU_DELTA.json').write_text(json.dumps(dict(host=h,remote=delta,local=local,equal=delta==local,reused_preflight=h+'_continuation_preflight.json'),indent=2));assert delta==local
tried=set()
for attempt in range(1,4):
 ss=statuses()
 shape=route[tuple(row['key'])]['manifest'];input_bytes=shape['B']*shape['N_kv']*(shape['D']+shape['PE'])*2
 eligible=[s for s in ss if s['host'] not in tried and s.get('free_disk_bytes',0)>input_bytes*2+3*1024**3 and s.get('rc') in (0,75)]
 eligible.sort(key=lambda s:(s['rc']!=0,s['host']=='local',-s.get('free_disk_bytes',0)))
 if not eligible:break
 host=eligible[0]['host'];tried.add(host)
 (P/'storage_decisions'/('sla-'+stamp+'-'+str(attempt)+'.json')).write_text(json.dumps(dict(key=row['key'],input_bytes=input_bytes,chosen_host=host,hosts=ss),indent=2))
 rc=process(row,host,attempt)
 if rc!=75:break
heartbeat(row['key'],'sla_attempt_finished','RESULT.json updated')
