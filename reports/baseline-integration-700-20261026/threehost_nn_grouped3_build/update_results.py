from pathlib import Path
import json,datetime
P=Path(__file__).resolve().parent
attempts={'moe-fwd-bfloat16':'local_bf16_01','moe-fwd-float16':'local_fp16_01','moe-dgrad-float16':'local_dgrad_01'}
def read(f):
 return json.loads(f.read_text()) if f.exists() else {}
def stamp(s):
 return datetime.datetime.fromisoformat(s).timestamp() if s else None
rows=[]
for s in read(P/'SCOPE.json'):
 cid=s['key'][2];a=P/attempts[cid];g=read(a/'GUARD_RESULT.json');c=read(a/'checker.json');m=read(a/'GUARD_METADATA.json');end=read(a/'GUARD_FINAL.json');f=read(a/'FROZEN.json')
 stages=[json.loads(l) for l in (a/'STAGES.jsonl').read_text().splitlines()] if (a/'STAGES.jsonl').exists() else []
 times={}
 for stage in ['reference','candidate']:
  start=next((r['at'] for r in stages if r['stage']==stage and r['event']=='started'),None);stop=next((r['at'] for r in stages if r['stage']==stage and r['event']=='returned'),None);times[stage+'_s']=stop-start if start and stop else None
 requested=stamp(m.get('requested_at'));acquired=stamp(m.get('acquired_at'));child=stamp(m.get('updated_at'));finished=stamp(end.get('updated_at')) if end.get('owner')=='baseline700-20261026/threehost_nn_grouped3_build' else None
 times.update(queue_s=acquired-requested if acquired and requested else None,physical_idle_wait_s=child-acquired if acquired and child else None,held_s=finished-acquired if finished and acquired else None,device_execution_s=None,wall_s=g.get('elapsed_s'))
 status='verified_pending_parent_review' if f else ('checker_'+c.get('status','unknown') if c else ('guard_not_completed' if a.exists() else 'not_run'))
 if g.get('returncode')==75:status='not_started_guard_wait_timeout'
 if g.get('returncode')==76:status='interrupted_external_context_no_verdict'
 if g.get('returncode')==124:status='guard_runtime_timeout_no_verdict'
 rows.append(dict(key=s['key'],status=status,numerical_verdict=True if f else None,registered=False,accepted_added=0,attempt=str(a),guard_exit=g.get('returncode'),requested_at=g.get('requested_at'),acquired_at=m.get('acquired_at'),child_start_at=m.get('updated_at'),end_at=g.get('finished_at'),timings=times,verification=c.get('verification'),input_seed=c.get('input_seed'),inputs_preserved=c.get('inputs_preserved'),evidence=str(a/'FROZEN.json') if f else None))
result=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted_added=0,verified_added=sum(r['numerical_verdict'] is True for r in rows),results=rows,remaining=[r['key'] for r in rows if r['numerical_verdict'] is not True],time_note='candidate time includes JIT and synchronization; exact device-only execution unavailable/null. Reference is original CPU checker under guard.')
(P/'RESULT.json').write_text(json.dumps(result,indent=2));print([(r['key'][2],r['status']) for r in rows])
