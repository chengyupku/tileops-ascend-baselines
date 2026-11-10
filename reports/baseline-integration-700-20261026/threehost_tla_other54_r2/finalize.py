import pathlib,json,time,datetime,hashlib,collections,re,ast
P=pathlib.Path(__file__).resolve().parent;E=P/'server1_evidence';r=json.loads((P/'RESULT.json').read_text());receipts=[(f,json.loads(f.read_text())) for f in P.glob('*.guard.json')]
def ts(v):return datetime.datetime.fromisoformat(v).timestamp()
def nels(x):
 if isinstance(x,dict):
  if x.get('kind')=='tensor':return x.get('elements',0)
  return sum(nels(v) for v in x.values())
 if isinstance(x,list):return sum(map(nels,x))
 return 0
checks=[]
for row in r['rows']:
 key=[row['baseline'],row['operator'],row['case_id']];attempts=[dict(receipt=f.name,**d) for f,d in receipts if d['key']==key];row['attempts_this_worker']=attempts
 matches=[]
 for f in (E/'runs').glob('*/checker.json'):
  c=json.loads(f.read_text())
  if [c['library'],c['op'],c['case_id']]==key:matches.append((f,c))
 if matches:
  f,c=matches[-1];row['checker_receipt']=str(f.relative_to(P));row['guard_receipt']=attempts[-1]['receipt'];row['native_evidence_directory']=str(f.parent.relative_to(P));row['dependency_hash_manifest']='server1_evidence/DEPENDENCY_ARTIFACT_HASHES.json';row['verification']=c.get('verification');row['observed_calls']=c.get('observed_calls');row['case_snapshot']=c.get('case_snapshot');row['input_seed']=c.get('input_seed');row['api']=c.get('api');row['source_sha256']=c.get('source_sha256');row['verifier_sha256']=c.get('verifier_sha256');row['oracle_source_sha256']=c.get('oracle_source_sha256')
  g=json.loads((f.parent/'GUARD_RUNNING.json').read_text());row['timings']=dict(reservation_queue_seconds=ts(g['acquired_at'])-ts(g['requested_at']),physical_idle_wait_seconds=ts(g['updated_at'])-ts(g['acquired_at']),checker_build_and_verify_seconds=c['wall_seconds'],guard_transport_wall_seconds=attempts[-1]['finished']-attempts[-1]['started'],compile_seconds=None,device_seconds=None,note='Checker includes compilation and verification; pure device duration not measured. Host clocks differ, so durations use same-host timestamp pairs only.')
  for nf in f.parent.glob('native_load*json'):
   d=json.loads(nf.read_text());artifact=f.parent/'native_artifacts'/d['sha256'];assert hashlib.sha256(artifact.read_bytes()).hexdigest()==d['sha256']
   for symbol,ptr in d['symbols'].items():
    assert any(int(l.split()[0].split('-')[0],16)<=ptr<int(l.split()[0].split('-')[1],16) and d['path'] in l for l in d['maps'].splitlines())
  checks.append(dict(key=key,status=c['status'],inputs_preserved=c['inputs_preserved'],outputs=c.get('verification'),source_unchanged=not c['provider_source_changed'] and not c['oracle_source_changed'],native_complete=row.get('native_evidence_complete',False)))
 row['accepted_added']=0;row['registered']=False
r.update(phase='completed_with_blockers',finished=time.time(),accepted_added=0,numerically_verified_this_worker=6,native_complete_verified=3,native_gap_verified=3,status_counts=dict(collections.Counter(x['status'] for x in r['rows'])),repeated_success_device_runs=0)
(P/'RESULT.json').write_text(json.dumps(r,indent=2))
h=json.loads((P/'HOST_ROUTING.json').read_text());h['device_attempts']=[d for f,d in receipts];(P/'HOST_ROUTING.json').write_text(json.dumps(h,indent=2))
for f in P.glob('*.py'):ast.parse(f.read_text(),filename=str(f))
check=dict(scope_exact_keys=len({(x['baseline'],x['operator'],x['case_id']) for x in r['rows']}),guard_attempts=len(receipts),all_attempts_in_scope=all(tuple(d['key']) in {(x['baseline'],x['operator'],x['case_id']) for x in r['rows']} for f,d in receipts),checks=checks,dependency_hashes=len(json.loads((E/'DEPENDENCY_ARTIFACT_HASHES.json').read_text())['files']),python_syntax='passed',shared_writes=False)
(P/'CHECKS.json').write_text(json.dumps(check,indent=2))
delivery=dict(status='completed_with_blockers',scope_count=54,numerical_passes=6,ready_for_native_evidence_review=3,numerical_passes_with_native_gap=3,blocked=48,accepted_added=0,registered=0,recovery='Use existing SDK9.2 beta.2 ARM compiler with official asc-devkit a63d5261c7c8077a0081e867f43a0e884fec066b release header view. No kernel/compiler/target edits.',verified_keys=[c['key'] for c in checks],blocking_groups=dict(original_default_architecture=37,All_original_semantics_or_backend=4,CausalConv_original_PTO_runtime_timeout=4,Sigmoid_no_complete_Ascend_API=1,NSA_original_FP16_signature_for_BF16_case=1,MHC_original_DMA_cast_semantic_rejection=1),native_gap='First three BMM passes have canonical checker, input/output records, generated cached artifacts and live maps. Actual symbol pointers and temporary loaded exported.so bytes were not captured before process exit. Do not treat these as fully closed native provenance; successful cases were not rerun.',actual_runtime_timeout='CausalConv original declared PTO compiled and entered native call with pointer/binary evidence, then guard exit124 after 300s; no numerical receipt. Other three same-route cases not executed.',artifacts=['RESULT.json','HOST_ROUTING.json','CHECKS.json','REGISTRATION_PROPOSAL.json','integration700_threehost_tla_other54.py','server1_evidence.tar.gz','server1_evidence/DEPENDENCY_ARTIFACT_HASHES.json'],archive_sha256=hashlib.sha256((P/'server1_evidence.tar.gz').read_bytes()).hexdigest(),restrictions='Only local task directory and matching server1 private directory written; no shared registry, memory, parent state, kernel or compiler changes. No child agents and no manual parent notification.',setup_issues='Three guard runs stopped before operator execution due to missing private snapshot layout/assets. Receipts preserved; subsequent runs used corrected frozen layout.',timing_note='RESULT separates guard queue and physical idle waits where metadata exists; no performance claims.')
(P/'DELIVERY.json').write_text(json.dumps(delivery,indent=2));(P/'HEARTBEAT.json').write_text(json.dumps(dict(phase='completed_with_blockers',currentkey=None,host='server1',lastprogress=time.time(),blockedreason='48 unresolved exact keys; 3 of 6 numerical passes retain native provenance gap'),indent=2));print(json.dumps({k:delivery[k] for k in ['scope_count','numerical_passes','ready_for_native_evidence_review','blocked','accepted_added']}))
