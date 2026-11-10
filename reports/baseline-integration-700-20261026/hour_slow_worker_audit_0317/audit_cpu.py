import pathlib,json,datetime,sqlite3,collections,re,hashlib
B=pathlib.Path('/root/dyq/workspace/reports/baseline-integration-700-20261026'); O=B/'hour_slow_worker_audit_0317'
ids={'3591f809e0574104':'threehost_linear49','f1ecbd644c6d4324':'threehost_attention_moe53','ab548181e0814aa3':'threehost_nn11','dd2a166c28284b81':'threehost_pto_original23'}
def epoch(v):
 if v is None:return None
 if isinstance(v,(float,int)):return float(v)
 return datetime.datetime.fromisoformat(v.replace('Z','+00:00')).timestamp()
def iso(v):return datetime.datetime.fromtimestamp(v,datetime.timezone.utc).isoformat() if v is not None else None
S=epoch('2026-10-27T02:17:42Z');E=epoch('2026-10-27T03:17:42Z')
def overlap(a,b):return max(0,min(b,E)-max(a,S)) if a is not None and b is not None else None
def rel(p):return str(p.relative_to(B)) if p.is_relative_to(B) else str(p)
def union(iv):
 out=[]
 for a,b in sorted(iv):
  if not out or a>out[-1][1]:out.append([a,b])
  else:out[-1][1]=max(b,out[-1][1])
 return sum(b-a for a,b in out)
c=sqlite3.connect('file:/root/dyq/codex-workers/workers.sqlite?mode=ro',uri=True)
A={'window':{'start':iso(S),'end':iso(E)},'accepted_added':0,'methodology':['CPU stdlib only; no case execution; no registry or math review','Worker lifetime from read-only workers.sqlite jobs.data created_at/finished_at; includes startup and shutdown','events.jsonl lacks timestamps: command/output counts are lifetime, fixed-window counts unknown','Successful fullcheck means checker status verified, checked_outputs=total_outputs>0, guard0; this is not formal acceptance','Attempt overlap is clipped to fixed window; sums are work-seconds, not elapsed walltime; union provided separately','Only task-local guard receipts, excluding prior_evidence; PTO RESULT/DELIVERY fields projected, no full dump'], 'workers':[]}
for id,name in ids.items():
 p=B/name; j=json.loads(c.execute('select data from jobs where id=?',(id,)).fetchone()[0]); ep=pathlib.Path(j['directory'])/'events.jsonl'; commands=[]
 for ln,l in enumerate(ep.open(),1):
  e=json.loads(l);it=e.get('item',{})
  if e.get('type')=='item.completed' and it.get('type')=='command_execution':commands.append({'line':ln,'command':it.get('command',''),'output_characters':len(it.get('aggregated_output','')),'exit_code':it.get('exit_code'),'output':it.get('aggregated_output','')})
 w={'id':id,'task':name,'lifetime':{'start':iso(j['created_at']),'end':iso(j['finished_at']),'seconds':j['finished_at']-j['created_at'],'window_overlap_seconds':overlap(j['created_at'],j['finished_at']),'source':'/root/dyq/codex-workers/workers.sqlite:jobs.data id='+id},'events':{'path':str(ep),'commands':len(commands),'output_characters':sum(x['output_characters'] for x in commands),'nonzero_commands':sum(x['exit_code'] not in (None,0) for x in commands),'window_commands':None,'window_output_characters':None,'timestamps':'unknown; raw events have no timestamps','largest_outputs':[{k:v for k,v in x.items() if k!='output'} for x in sorted(commands,key=lambda x:x['output_characters'],reverse=True)[:5]],'profiling_transfer_command_samples':[{k:v for k,v in x.items() if k!='output'} for x in commands if re.search('setprofile|set_num_threads|migrate_runtime|FULL_OPP_TRANSFER|cancel_own_for_threads',x['command'])][:12]},'attempts':[]}
 pto={}
 if name=='threehost_pto_original23':
  result=json.loads((p/'RESULT.json').read_text())
  for row in result['rows']:
   for at in row.get('attempts',[]):pto[at['guard']]=at.get('run')
  w['selected_receipts']=[{k:r.get(k) for k in ['key','status','host','checker','guard']} for r in result['rows']]
  w['pto_exclusion_note']=json.loads((p/'DELIVERY.json').read_text()).get('excluded_attempt_note')
 guards=[f for f in p.rglob('*.json') if (f.name in ['GUARD_RESULT.json','guard.json'] or f.name.endswith('.guard.json')) and 'prior_evidence' not in str(f)]
 for g in guards:
  x=json.loads(g.read_text());a=epoch(x.get('started_epoch',x.get('started',x.get('requested_at'))));b=epoch(x.get('finished_epoch',x.get('finished',x.get('finished_at',x.get('updated_at')))));rc=x.get('returncode',x.get('exit_code'));m=x.get('metadata') or {};a=a if a is not None else epoch(m.get('requested_at'));b=b if b is not None else epoch(m.get('updated_at'));cp=g.parent/'checker.json'
  if str(g) in pto:cp=pathlib.Path(pto[str(g)])/'checker.json'
  elif name=='threehost_pto_original23':
   stem=g.name.removesuffix('.guard.json').removeprefix('tilelang-ascend_');cand=[p/h/'runs'/stem/'checker.json' for h in ['local_evidence','server1_evidence','server2_evidence']];cp=next((f for f in cand if f.exists()),cp)
  ck=json.loads(cp.read_text()) if cp.exists() else {};v=ck.get('verification') or {}; ok=rc==0 and ck.get('status')=='verified' and v.get('checked_outputs',0)>0 and v.get('checked_outputs')==v.get('total_outputs')
  r={'guard':rel(g),'start':iso(a),'end':iso(b),'wall_seconds':x.get('wall_seconds',x.get('elapsed_s',b-a if a is not None and b is not None else None)),'window_overlap_seconds':overlap(a,b),'exit':rc,'checker':rel(cp) if cp.exists() else None,'checker_status':ck.get('status'),'checker_error':ck.get('error'),'verification':v,'fullcheck_pass':ok,'key':[ck.get('library'),ck.get('op'),ck.get('case_id')],'checker_wall_seconds':ck.get('wall_seconds')}
  logs=[g.with_suffix('.log'),g.parent/'guard.log',g.parent/'stderr.log'];samples=[]
  for log in logs:
   if log.exists():
    for ln,l in enumerate(log.open(errors='replace'),1):
     if re.search('external Ascend|status.*verified|Traceback|Error:|returned [0-9]|VERSION|setprofile',l):samples.append({'path':rel(log),'line':ln,'excerpt':l[:450].strip()})
    if samples:break
  r['log_samples']=samples[:4];w['attempts'].append(r)
 for label,rs in [('lifetime',w['attempts']),('window', [r for r in w['attempts'] if (r['window_overlap_seconds'] or 0)>0])]:
  w[label+'_attempt_summary']={'attempts':len(rs),'exit_counts':dict(collections.Counter(str(r['exit']) for r in rs)),'fullcheck_pass_receipts':sum(r['fullcheck_pass'] for r in rs),'unique_pass_keys':len({tuple(r['key']) for r in rs if r['fullcheck_pass']}),'guard_work_seconds_by_exit':{str(code):sum((r['wall_seconds'] if label=='lifetime' else r['window_overlap_seconds']) or 0 for r in rs if r['exit']==code) for code in {r['exit'] for r in rs}},'guard_interval_union_seconds':union([(max(epoch(r['start']),S) if label=='window' else epoch(r['start']),min(epoch(r['end']),E) if label=='window' else epoch(r['end'])) for r in rs if r['start'] and r['end']])}
 A['workers'].append(w)
(O/'AUDIT.json').write_text(json.dumps(A,ensure_ascii=False,indent=2)+'\n')
for w in A['workers']:
 print(w['task'],w['lifetime'],w['lifetime_attempt_summary'],w['window_attempt_summary'])
