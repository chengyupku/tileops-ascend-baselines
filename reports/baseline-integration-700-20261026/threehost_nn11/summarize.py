from pathlib import Path
import json,datetime,sys
P=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text()) if p.exists() else {}
def dt(s):return datetime.datetime.fromisoformat(s).timestamp()
rows=[]
for old in read(P/'SCOPE.json'):
 key=[old['baseline'],old['operator'],old['case_id']];hist=[]
 for g in P.glob('**/GUARD_RESULT.json'):
  if 'snapshot' in g.parts:continue
  q=g.parent;cmd=read(q/'COMMAND.json')
  if not cmd or cmd[-2]!=key[2] or key[1]!='GroupedGemmFwdOp':continue
  guard=read(g);c=read(q/'checker.json');m=read(q/'GUARD_METADATA.json');final=read(q/'GUARD_FINAL.json')
  own=m or (final if final.get('owner')=='baseline700-20261026/threehost_nn11' and final.get('requested_at','')>=guard.get('requested_at','') else {})
  timings={'wall_s':guard['elapsed_s'],'queue_s':None,'physical_idle_wait_s':None,'held_s':None}
  if own and all(x in own for x in ['requested_at','acquired_at']):timings['queue_s']=dt(own['acquired_at'])-dt(own['requested_at'])
  if m:timings['physical_idle_wait_s']=dt(m['updated_at'])-dt(m['acquired_at'])
  if m and final.get('owner')==m.get('owner') and final.get('pid')==m.get('pid'):timings['held_s']=dt(final['updated_at'])-dt(m['updated_at'])
  if guard['returncode']==75:timings['reservation_wait_limit_s']=120;timings['physical_idle_wait_s']=None
  stages=[]
  if (q/'STAGES.jsonl').exists():stages=[json.loads(x) for x in (q/'STAGES.jsonl').read_text().splitlines()]
  for label in ['reference','candidate']:
   events=[e for e in stages if e['stage']==label]
   if len(events)==2:timings[label+'_s']=events[1]['at']-events[0]['at']
  hist.append(dict(at=guard['requested_at'],host='server1' if any('server1' in x for x in q.relative_to(P).parts) else 'server2' if any('server2' in x for x in q.relative_to(P).parts) or q.name.startswith('arm_') else 'local',attempt=str(q),guard=str(g),guard_returncode=guard['returncode'],checker=str(q/'checker.json') if c else None,checker_status=c.get('status'),checker_stage=c.get('stage'),error=c.get('error'),timings=timings,verification=c.get('verification'),native=str(q/'native.jsonl') if (q/'native.jsonl').exists() else None,frozen=str(q/'FROZEN.json') if (q/'FROZEN.json').exists() else None))
 hist.sort(key=lambda x:x['at'])
 status=old['status'];reason=old['reason']
 if hist:
  last=hist[-1]
  if last['guard_returncode']==0 and last['checker_status']=='verified':status='verified_pending_parent_review';reason='Original complete checker and clean guard succeeded; parent acceptance remains separate.'
  elif last['guard_returncode']==75:status='blocked_reservation_wait';reason='No case execution; cooperative reservation wait exceeded120s. Other hosts checked and deployment prepared.'
  elif last['guard_returncode']==76:status='interrupted_external_occupancy';reason='External device context detected; guard terminated this attempt. No numerical verdict.'
  else:status='environment_attempt_failed';reason=last['error'] or 'See retained guarded import/runtime exception. Private relocation fixes recorded.'
 rows.append(dict(key=key,status=status,reason=reason,prior_status=old['status'],prior_checker=old.get('checker_receipt'),attempts=hist,accepted_added=0,registered=False))
result=dict(status='complete_delivery' if '--final' in sys.argv else 'in_progress',updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope_size=len(rows),verified=sum(r['status']=='verified_pending_parent_review' for r in rows),accepted_added=0,rows=rows,host_routing=str(P/'HOST_ROUTING.json'),remaining_diagnosis=str(P/'REMAINING_DIAGNOSIS.json'),note='CLI completion is not numerical success or parent acceptance. Remote clocks differ; durations use same-host timestamps only.')
(P/'RESULT.json').write_text(json.dumps(result,indent=2));print('verified',result['verified'],'scope',len(rows))
