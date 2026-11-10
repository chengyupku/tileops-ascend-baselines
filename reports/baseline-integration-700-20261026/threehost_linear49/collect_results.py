from pathlib import Path
import json,hashlib,datetime
P=Path(__file__).resolve().parent;rows=json.loads((P/'SCOPE.json').read_text());by={(x['baseline'],x['operator'],x['case_id']):x for x in rows}
for x in rows:
 x['prior_scope_status']=x['status'];x['executed_this_task']=False;x['accepted_added']=0;x['registered']=False;x['attempts']=[]
prior=json.loads((P.parent/'gdn_fp16_capture14/RESULT.json').read_text())
for x in prior:
 key=(x['baseline'],x['operator'],x['case_id'])
 if key in by:
  r=by[key];r['latest_prior_evidence']='../gdn_fp16_capture14/RESULT.json';r['reason']=x['reason'];r['status']='failed' if x['status']=='diagnostic_failed' else 'blocked';r['prior_latest_status']=x['status'];r['prior_latest_checker']=str(P.parent/'gdn_fp16_capture14'/x['checker_receipt']) if x.get('checker_receipt') else None
for f in sorted(P.glob('**/GUARD_RESULT.json'),key=lambda x:x.stat().st_mtime):
 if 'snapshot' in f.parts:continue
 d=f.parent;g=json.loads(f.read_text());c=json.loads((d/'checker.json').read_text()) if (d/'checker.json').is_file() else None
 if c:key=(c['library'],c['op'],c['case_id'])
 else:
  a=g['argv']
  if '--case-id' in a:key=(a[a.index('--library')+1],a[a.index('--op')+1],a[a.index('--case-id')+1])
  elif d.name.startswith('fla_'):
   matches=[k for k in by if k[0]=='flash-linear-attention-npu' and k[1]=='GDNFwdOp' and (d.name=='fla_'+k[2] or d.name.startswith('fla_'+k[2]+'_'))]
   assert len(matches)==1,(d,matches)
   key=matches[0]
  else:continue
 if key not in by:raise RuntimeError(key)
 r=by[key];rc=g['returncode'];m=g.get('metadata',{});times={'guard_total_seconds':g.get('wall_seconds'),'queue_seconds':None,'physical_wait_seconds':None,'checker_seconds':c.get('wall_seconds') if c else None,'compile_seconds':None,'pure_device_seconds':None,'native_call_wall_seconds':None}
 if (d/'dispatch.jsonl').exists():times['native_call_wall_seconds']=sum(json.loads(line).get('wall_seconds',0) for line in (d/'dispatch.jsonl').read_text().splitlines())
 if m.get('owner')=='baseline700-20261026/threehost_linear49' and m.get('requested_at') and m.get('acquired_at'):
  times['queue_seconds']=(datetime.datetime.fromisoformat(m['acquired_at'])-datetime.datetime.fromisoformat(m['requested_at'])).total_seconds()
 if (d/'GUARD_RUNNING.json').exists():
  z=json.loads((d/'GUARD_RUNNING.json').read_text());times['physical_wait_seconds']=(datetime.datetime.fromisoformat(z['updated_at'])-datetime.datetime.fromisoformat(z['acquired_at'])).total_seconds()
 att={'directory':str(d.relative_to(P)),'guard_receipt':str(f.relative_to(P)),'guard_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'guard_returncode':rc,'checker_receipt':str((d/'checker.json').relative_to(P)) if c else None,'timings':times,'native_evidence':[str(q.relative_to(P)) for q in [d/'dispatch.jsonl',d/'maps.txt',d/'NATIVE_IDENTITY.json',d/'LOADED_HASHES.json'] if q.exists()]}
 att['checker_status']=c.get('status') if c else None;att['checker_error']=c.get('error') if c else None
 r.pop('teardown_issue',None)
 r['attempts'].append(att);r['timings']=times;r['guard_receipt']=att['guard_receipt'];r['checker_receipt']=att['checker_receipt'];r['executed_this_task']=r['executed_this_task'] or c is not None
 if rc in (75,76,78):r['status']='queued_not_started' if rc==75 else 'guard_blocked';r['reason']='Guard '+str(rc)+'; not a numerical failure. See host-routing receipts.'
 elif c:
  r['status']='verified' if c.get('status') in ('passed','verified') and rc==0 else 'failed' if 'Tensor-likes are not close' in c.get('error','') else 'blocked';r['reason']=c.get('error',c.get('status'));r['checker_status']=c.get('status');r['checker_stage']=c.get('stage');r['observed_calls']=c.get('observed_calls');r['input_seed']=c.get('input_seed');r['inputs_preserved']=c.get('inputs_preserved',False);r['source_hash']=c.get('source_sha256',{});r['oracle_source_sha256']=c.get('oracle_source_sha256',{});r['checked_outputs']=2 if r['status']=='verified' else 0
  if rc==134:r['teardown_issue']='free(): invalid pointer after checker; guard rc134; cannot accept'
 else:r['status']='blocked';r['reason']='Process ended without canonical checker receipt; guard rc'+str(rc)
(P/'RESULT.json').write_text(json.dumps(rows,indent=2));print({k:sum(r['status']==k for r in rows) for k in set(r['status'] for r in rows)})
