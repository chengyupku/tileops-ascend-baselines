"""CPU-only evidence assertions and short, locked acceptance publication."""
import ast, datetime, fcntl, hashlib, json, pathlib, shutil
W=pathlib.Path('/root/dyq/workspace'); R=W/'reports/baseline-integration-700-20261026'; A=R/'acceptance_root'; L=W/'reports/baseline-single-agent-local-20261027'
def read(p): return json.loads(pathlib.Path(p).read_text())
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def put(p,d):
 p=pathlib.Path(p); t=p.with_suffix(p.suffix+'.tmp.2c16f31f');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
now=datetime.datetime.now(datetime.timezone.utc).isoformat(); owner='2c16f31fa2744933'
q=read(R/'non_tla_transformer79/PUBLISH_QUEUE.json')['entries']; assert len(q)==7
review=read(R/'non_tla_transformer79/fp8_local_evidence7/REVIEW.json')
pins=read(L/'FP8_RUNTIME_PREFLIGHT.json')['source_sha256']
for p,h in pins.items(): assert sha(p)==h,p
for section in ['semantic_review','registration']:
 for v in review[section].values():
  if isinstance(v,dict) and 'path' in v and 'sha256' in v: assert sha(v['path'])==v['sha256'],v['path']
identity=q[0]['source_identity']
for v in identity.values():
 if isinstance(v,dict) and 'path' in v and 'sha256' in v: assert sha(v['path'])==v['sha256'],v['path']
for p,h in identity['reused_sdk_hashes_current_match'].items(): assert sha(p)==h,p
reg=read(L/'FP8_REGISTRATION.json');assert sha(reg['registry'])==reg['after_sha256']
assert '_install_single_local_fp8(PROVIDERS)' in pathlib.Path(reg['registry']).read_text()
for p in [reg['registry'],reg['provider'],reg['adapter'],L/'fp8_registration.py',L/'run_fp8_padded.py']: ast.parse(pathlib.Path(p).read_text())
verified={tuple(x['key']):x for x in read(L/'FP8_VERIFIED_CASES.json')}
rows=[]
for e in q:
 key=e['key']; c=read(e['checker']); g=read(e['guard']); n=read(e['native']); run=read(e['execution']); gm=g['guard_metadata']
 assert [c['library'],c['op'],c['case_id']]==key
 assert c['status']=='verified' and c['inputs_preserved'] and not c['oracle_source_changed'] and not c['provider_source_changed']
 assert c['verification']['checked_outputs']==c['verification']['total_outputs']==1 and c['verification']['unchecked_reason'] is None
 assert c['input_seed']==e['input_seed'] and all(c['source_sha256'].get(p)==h for p,h in pins.items())
 assert c['verifier_sha256']==sha(W/'TileOPs/scripts/ascend_pair_check.py')
 for p,h in c['oracle_source_sha256'].items(): assert sha(W/p)==h,p
 assert c['case_snapshot']==verified[tuple(key)]['snapshot']
 assert g['returncode']==0 and gm['workload_completed'] and gm['workload_exit_code']==0 and gm['phase']=='finished'
 assert gm['performance_valid'] and not gm['conflict_detected'] and not gm['visibility_lost']
 for k in ['pid','start_ticks','owner','child','boot_id']: assert run[k]==gm[k]
 assert gm['owner']=='baseline-single-agent-local-20261027/'+pathlib.Path(e['checker']).parent.name
 assert g['started_at']<=c['started_at']<c['finished_at']<=g['ended_at']
 assert len(n)==1 and n[0]['api']=='aclnnGroupedMatmulV5' and n[0]['native_calls']==1 and n[0]['synchronized'] and n[0]['padding_factor']==128
 assert n[0]['binary'] in pins and n[0]['binary'] in pathlib.Path(e['loaded_maps']).read_text()
 params=c['case_snapshot']['parameters'];m,k=params['a'][0];nn=params['b'][0][0]; assert n[0]['swapped']==(m<=nn)
 lhs,cols=(nn,m) if m<=nn else (m,nn)
 assert n[0]['native_shapes']==[[lhs,k],[1,k,cols*128],[lhs,k//128],[1,k//128,cols],[lhs,cols*128]]
 calls=c['observed_calls']['candidate'];assert calls and all(x['returned'] and x['outputs']['shape']==[m,nn] for x in calls)
 rows.append({'key':key,'decision':'ACCEPT','evidence_review_completed':True,'host':'local','checker':e['checker'],'guard':e['guard'],'native':e['native'],'verification':c['verification'],'evidence_sha256':{k:sha(e[k]) for k in ['checker','guard','native','execution','loaded_maps']}})
report={'at':now,'owner':owner,'rows':rows,'runtime_pins_checked':len(pins),'semantic_review':review['semantic_review'],'registration':review['registration'],'source_identity':identity,'provenance':'Reused user local agent success; not remote execution throughput','remote_executions':0,'device_reruns':0,'performance_claim':False,'limitation':'128x padded output and local vendor launcher required; all mapping copies inside run','actual_accepted_added':0}
put(A/'REVIEW_LOCAL7.json',report)
with open(reg['lock'],'a') as pl,open(R/'ACCEPTED.lock','a') as al,open(R/'ACCEPTED.json.lock','a') as legacy:
 for f in [pl,al,legacy]: fcntl.flock(f,fcntl.LOCK_EX)
 assert sha(reg['registry'])==reg['after_sha256']
 ledger=read(R/'ACCEPTED.json'); keys={tuple(x['key']) for x in ledger['rows']}; before=len(keys)
 shutil.copy2(R/'ACCEPTED.json',A/'ACCEPTED.before_2c16f31fa2744933.json')
 for row in rows:
  if tuple(row['key']) in keys: continue
  ledger['rows'].append({'key':row['key'],'at':now,'accepted':True,'host':'local','checker':row['checker'],'guard':row['guard'],'native':row['native'],'parent_review':str(A/'REVIEW_LOCAL7.json'),'qualification':'Original complete aclnnGroupedMatmulV5 with semantically equivalent zero-row padding/layout mapping; original full checker and input preservation, clean guard and pinned native identity. Reused user local agent evidence; no remote throughput or performance claim; no rerun.'}); keys.add(tuple(row['key']))
 ledger['updated_at']=now;put(R/'ACCEPTED.json',ledger)
 report.update(actual_accepted_added=len(keys)-before,accepted_total=len(keys));put(A/'REVIEW_LOCAL7.json',report)
put(A/'HEARTBEAT.json',{'at':now,'owner':owner,'actual_accepted_added':report['actual_accepted_added'],'pending':0,'status':'reviewed_local7_monitoring_non_tla_groups','blockers':[]})
print(json.dumps({'reviewed':len(rows),'actual_accepted_added':report['actual_accepted_added'],'accepted_total':len(keys)}))
