from pathlib import Path
import json,datetime,collections,hashlib
P=Path(__file__).resolve().parent;now=datetime.datetime.now(datetime.timezone.utc).isoformat();rows=json.loads((P/'RESULT.json').read_text());audit=json.loads((P/'VERIFIED_AUDIT.json').read_text())
for r in rows:
 if r['baseline']=='ops-transformer' and r['operator']=='DeltaNetFwdOp' and r['case_id']=='prefill-12k-continued-bfloat16':
  r['status']='not_run';r['reason']='Metadata deployment blocker resolved in this task; this exact case not executed. No inference from other passing sizes.'
(P/'RESULT.json').write_text(json.dumps(rows,indent=2));counts=dict(collections.Counter(r['status'] for r in rows));verified=[r for r in rows if r['status']=='verified'];assert len(verified)==len(audit)
handoff=[]
for r in verified:
 path=r['checker_receipt'];package='package_local' if not path.startswith('server1/') else ('server1/package_recurrent' if 'recurrent91' in path else 'server1/package_packed' if 'varlen' in path else 'server1/package')
 handoff.append({'exactkey':[r['baseline'],r['operator'],r['case_id']],'provider_package':package,'checker':path,'guard':r['guard_receipt'],'audit':str(Path(path).parent/'VALIDATION_AUDIT.json'),'requires_parent_review':True})
(P/'REGISTRATION_HANDOFF.json').write_text(json.dumps({'accepted_added':0,'registered_added':0,'entries':handoff,'warning':'Use actual collected server1 packages, not initial local package seed. ABI-specific recipes are pinned. Do not use abandoned runtime92 partial transfer.'},indent=2))
checks={'utc':now,'scope_count':len(rows),'unique_scope_keys':len({(r['baseline'],r['operator'],r['case_id']) for r in rows}),'verified_audit_count':len(audit),'all_verified_have_guard0_full_outputs_input_preservation':True,'local_post_integrity':json.loads((P/'POST_RUN_INTEGRITY.json').read_text())['passed'],'server1_post_integrity':json.loads((P/'server1/POST_RUN_INTEGRITY.json').read_text())['passed'],'no_numerical_success_inferred_for_other_keys':True,'accepted_added':0}
assert checks['scope_count']==checks['unique_scope_keys']==49 and checks['local_post_integrity'] and checks['server1_post_integrity']
(P/'FINAL_CHECKS.json').write_text(json.dumps(checks,indent=2))
delivery={'status':'partial_scope_verified_results_with_remaining_blocks','updated_at':now,'scope_count':49,'status_counts':counts,'verified_count':len(verified),'verified':handoff,'registered_added':0,'accepted_added':0,'result':'RESULT.json','host_routing':'HOST_ROUTING.json','verification_audit':'VERIFIED_AUDIT.json','checks':'FINAL_CHECKS.json','shared_registry_or_memory_modified':False,'remaining':'Per-key failures, unsupported APIs/reference paths, guard75 queue and guard76 invalidation retained; pending exact cases require future safe scheduling. Numerical verification is not parent acceptance.','provenance_limit':'Complete original operator source tree collected during task; native dependencies pinned before execution and rechecked after. Do not interpret later complete source collection as pre-run attestation of every source file.','timing_limit':'Queue/physical wait separated where receipts support them; unavailable stages remain null. Native call wall time includes overhead and is not pure device duration. Remote clock offset prevents cross-host timestamp subtraction.'}
(P/'DELIVERY.json').write_text(json.dumps(delivery,indent=2))
text='''# threehost_linear49 delivery

Scope: 49 exact keys; accepted_added=0, registered_added=0. Shared providers, original kernels/checkers, parent state and memory were not modified.

Actual numerical passes (original fullchecker, both outputs and input preservation):
'''+''.join('- '+r['case_id']+' (`'+r['checker_receipt']+'`)\n' for r in verified)+'''
Repairs: genuine SDK9.1 metadata/compiler environment; unchanged ARM ChunkGatedDeltaRule and RecurrentKda packages migrated into private server1 directory; descriptor allocator aligned with actual SDK9.1 native linkage, eliminating mixed-nnopbase teardown abort. Original full RecurrentKda API with zero log-gate supplies ungated decode and FP32 final state. Packed CPU-offset fallback fixes missing optional CPU offsets. Local genuine SDK9.2 metadata and original x86 package enable compatible migration. No kernel/compiler/architecture/reference/tolerance/seed changes.

Failures retained: complete original FP16 ChunkKdaFwd still fails DeltaNet decode (32736/32768 attention elements); original FLA recurrent API still fails beta-L2 decode final state (13/131072). Earlier numerical failures remain in attempt receipts. Old FP16 deployment-only diagnosis superseded by existing complete-API execution evidence, without rerunning prior probes. Unsupported FP64 reference paths were not rewritten.

Scheduling: initial server2 external occupancy triggered server1 migration; later server1/2 occupancy triggered local migration. Guard75 means not started; 16k guard76 means contaminated and invalid, not numerical failure. See HOST_ROUTING.json for all three host receipts and migration events.

Evidence: RESULT.json enumerates all exact keys and attempts; VERIFIED_AUDIT.json checks pointer/maps/inode plus device launch and original binary registration. Raw stat and maps device IDs retained separately (overlay filesystem). FINAL_CHECKS.json and POST_RUN_INTEGRITY.json record source/native hash rechecks. REGISTRATION_HANDOFF.json points to actual runtime packages. Initial local package seed and abandoned full runtime92 transfer are not the final provider. Parent must audit and register; no accepted keys are added here.
'''
(P/'REPORT.md').write_text(text)
hb={'phase':'finished_partial_scope_delivery','currentkey':None,'host':['local','server1','server2'],'lastprogress':now,'blockedreason':'Remaining exact keys retain numerical/API/reference blockers or require safe scheduling; see RESULT.json','verified_this_task':len(verified),'accepted_added':0}
for n in ['HEARTBEAT.json','HEARTBEAT.local.json']:(P/n).write_text(json.dumps(hb,indent=2))
manifest={}
for p in P.rglob('*'):
 if not p.is_file() or p.name=='ARTIFACT_MANIFEST.json' or p.suffix in ['.gz'] or 'runtime92' in p.parts:continue
 manifest[str(p.relative_to(P))]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
(P/'ARTIFACT_MANIFEST.json').write_text(json.dumps({'utc':now,'files':manifest,'excluded':'large abandoned/full runtime tarballs; unchanged original native packages individually pinned by manifests'},indent=2));print(counts,'artifacts',len(manifest))
