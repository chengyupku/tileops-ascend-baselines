import pathlib,json,datetime,hashlib,collections
P=pathlib.Path(__file__).resolve().parent
utc=datetime.datetime.now(datetime.timezone.utc).isoformat(); rows=[]
qpath=P/'PUBLISH_QUEUE.json'
q=json.loads(qpath.read_text()) if qpath.exists() else {'entries':[],'accepted_added':0}
q['last_review_utc']=utc
if not q.get('entries'):q['reason']='No reviewed original full-checker success with deployable native evidence; registration not ready'
qpath.write_text(json.dumps(q,ensure_ascii=False,indent=2))
for w in sorted(P.glob('worker[0-9][0-9]')):
 scope=json.loads((w/'SCOPE.json').read_text());f=w/'RESULT.json'
 try:x=json.loads(f.read_text())
 except:x=[]
 rr=x if isinstance(x,list) else x.get('exactkeys',x.get('results',x.get('rows',[])))
 for a in scope:
  key=a['key']; matches=[(i,r) for i,r in enumerate(rr) if r.get('key')==key]
  last=matches[-1][1] if matches else {}
  checker=last.get('checker')
  status=last.get('status','awaiting_worker_result')
  if isinstance(checker,dict) and checker.get('status')=='failed' and (checker.get('stage')=='build' or 'Unresolved call Op(' in str(checker.get('error',''))):status='original_compile_failed'
  rows.append({'key':key,'owner':w.name,'status':status,'worker_reported_status':last.get('status'),'numerical_success_claimed':any(r.get('numerical_success') is True for _,r in matches),'coordinator_verified':False,'registered':False,'accepted_added':0,'worker_result':str(f) if f.exists() else None,'matching_row_indices':[i for i,_ in matches],'evidence':last.get('evidence',last.get('prior_exact_evidence')),'reason':last.get('reason'),'timings':last.get('timings'),'pending':not matches or any(s in status for s in ['pending','not_run','unverified','not_started','guard_requests'])})
h=json.loads((P/'GROUP_HEARTBEAT.json').read_text());counts=dict(collections.Counter(r['status'] for r in rows))
result={'utc':utc,'status':'supervision_in_progress' if h['active'] else 'children_finished_pending_final_review','scope':97,'unique_keys':len(set(tuple(r['key']) for r in rows)),'accepted_added':0,'coordinator_verified':0,'numerical_success_claimed':sum(r['numerical_success_claimed'] for r in rows),'rows':rows,'status_counts':counts}
(P/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
delivery={'utc':utc,'status':result['status'],'accepted_added':0,'verified':0,'scope':97,'active':h['active'],'alive':h['alive'],'CLI_completed':h['CLI completed'],'root_accepted_read_only':h['root accepted'],'status_counts':counts,'publish_queue':'PUBLISH_QUEUE.json','evidence_index':'RESULT.json','process_samples':'PROCESS_SAMPLES.jsonl','replacements':'REPLACEMENTS.jsonl','checks':['97 exclusive scope keys','Native PID/start stamps sampled','Parent accepted scope intersection checked','Compile failures excluded from numerical verdicts','No coordinator NPU imports or device runs','Shared registry and task memory untouched'],'remaining':['Children still running; not a final completion claim'] if h['active'] else ['Review final per-key blockers and any new full-checker successes'],'timing_policy':'Host-clock guard timing only; absent fields null; aggregate controller wall is not device time.'}
(P/'DELIVERY.json').write_text(json.dumps(delivery,ensure_ascii=False,indent=2))
(P/'HEARTBEAT.json').write_text(json.dumps({'utc':utc,'phase':result['status'],'last_substantive_output':'RESULT.json exact-key aggregate and process identity samples','active':h['active'],'alive':h['alive'],'accepted_added':0,'next':'Continue child supervision; review any full-checker successes'},indent=2))
files=['monitor.py','aggregate.py','replace_worker.py','message_child.py','send_guidance.py','RESULT.json','DELIVERY.json','HEARTBEAT.json','PUBLISH_QUEUE.json','GROUP_REVIEW.json','COORDINATION.json','prepare_continuation.py','DEPLOYMENT_TYPEERROR.json','SCHEDULING_CORRECTION.json','ORPHAN_DRIVER_CONTROL.json']
(P/'FILEHASHES.json').write_text(json.dumps({'utc':utc,'sha256':{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in files if (P/n).exists()}},indent=2))
print(json.dumps({'keys':len(rows),'unique':result['unique_keys'],'claimed_pass':result['numerical_success_claimed'],'status_counts':counts}))
