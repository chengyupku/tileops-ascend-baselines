import pathlib,json,hashlib,datetime,collections,ast
P=pathlib.Path(__file__).resolve().parent
utc=datetime.datetime.now(datetime.timezone.utc).isoformat();result=json.loads((P/'RESULT.json').read_text());hb=json.loads((P/'GROUP_HEARTBEAT.json').read_text());pq=json.loads((P/'PUBLISH_QUEUE.json').read_text())
num={'numerical_failed','prior_exact_numerical_failure_preserved','failed_preserved','prior_numeric_failed_exact_key','numeric_failed_exact_key','prior_exact_checker_failed_no_retry','failed','canonical_checker_failed','prior_canonical_checker_failed_no_retry','original_numerical_failure'}
api={'original_complete_api_boundary','not_run_original_api_boundary','blocked_original_api_shape','blocked_no_complete_original_api','unsupported'}
def category(r):
 s=r['status']
 if s.startswith('verified'):return 'verified_pending_parent_registration'
 if s in num:return 'original_exact_numerical_failure'
 if s in api:return 'original_api_or_dtype_boundary'
 if s in ['oracle_blocked_before_candidate','blocked_original_oracle_dt_double']:return 'original_oracle_error_before_candidate'
 if s=='blocked_validator_none_coverage':return 'original_checker_coverage_failure'
 if s=='blocked_prior_device_execution_error':return 'prior_original_device_execution_error'
 if s=='not_run_exact_single_card_capacity_bound':return 'physical_capacity_unexecuted'
 return 'unresolved'
counts=collections.Counter(category(r) for r in result['rows']);assert sum(counts.values())==92;assert not counts['unresolved'];assert len(pq['entries'])==2
for e in pq['entries']:
 assert e['ready_to_publish'] and e['evidence_review_passed'];assert hashlib.sha256(pathlib.Path(e['code']).read_bytes()).hexdigest()==e['code_sha256'];assert hashlib.sha256(pathlib.Path(e['registration_pins']).read_bytes()).hexdigest()==e['registration_pins_sha256']
syntax=[]
for p in P.glob('*.py'):ast.parse(p.read_text());syntax.append(p.name)
checked={'utc':utc,'unique_exact_keys':len({tuple(r['key']) for r in result['rows']}),'scope_count':92,'unresolved_delivery_rows':0,'coordinator_python_syntax':syntax,'publication_entries':2,'module_and_pin_hashes_match':True,'original_evidence_review':'REVIEW_ds-v3-*.json','real_cpu_registry_route_check':'CPU_ROUTE_REVIEW.json','patch_check':'git apply --check PUBLISH_REGISTRY.patch passed; parent serial publication still required','npu_operations_by_coordinator':0,'accepted_added':0}
(P/'CHECKS.json').write_text(json.dumps(checked,ensure_ascii=False,indent=2))
replacements=[x for x in json.loads((P/'REPLACEMENT_ACTIONS.json').read_text()) if x.get('action')!='final_real_remaining_check'];replacements.append({'utc':utc,'action':'final_real_remaining_check','reason':'All92 keys have final exact results or original immutable constraints; 2 verified packages await parent serial publication. No legal unchanged numerical/API replay or unowned device-executable scope remains. Physical capacity blocker108.66GiB exceeds96GiB allhosts.'})
(P/'REPLACEMENT_ACTIONS.json').write_text(json.dumps(replacements,ensure_ascii=False,indent=2))
all_rows=[{'key':r['key'],'category':category(r),'status':r['status'],'evidence':r.get('group_result_source')} for r in result['rows']]
(P/'FINAL_KEY_SUMMARY.json').write_text(json.dumps({'utc':utc,'categories':counts,'rows':all_rows},ensure_ascii=False,indent=2))
delivery={'utc':utc,'group':'surge_dense_linear92','scope_count':92,'task_type':'delegated original-baseline integration and exact supplementary checking','coordination_cpu_only':True,'initial_direct_workers':7,'native_jobs_including_replacements':len(hb['workers']),'max_concurrent_direct_workers_observed':7,'model':'gpt-6-astra','effort':'medium','categories':dict(counts),'original_fullchecker_passed':2,'ready_to_publish':2,'accepted_added':0,'registered_by_group':False,'parent_root_accepted_observation':hb['root accepted'],'parent_group_accepted_observation':hb['group_root_accepted'],'global_rolling60m_accepted':hb['root_accepted_rolling60m'],'throughput_target_100_per_hour_met':hb['root_accepted_rolling60m']>=100,'process_identity_sample_survival':hb['process_identity_sample_survival'],'time_weighted_7slot_occupancy':hb['time_weighted_7slot_occupancy'],'native_capacity_failures':1,'intentional_own_worker_cancellations':1,'model_queue_seconds':None,'completion_status':'all executable scope handled;2 ready for parent serial publication;90 nonaccepted original constraints/failures','publish_queue':str(P/'PUBLISH_QUEUE.json'),'publication_patch':str(P/'PUBLISH_REGISTRY.patch'),'results':str(P/'RESULT.json'),'per_key_summary':str(P/'FINAL_KEY_SUMMARY.json'),'checks':str(P/'CHECKS.json'),'corrective_actions':str(P/'COORDINATION_OUTCOMES.json'),'pending_parent_steps':['Review2 exact success evidence already aggregated','Apply concrete module+registry patch serially and retain pinned dependency paths','Parent alone updates formal registry/ACCEPTED; group never writes shared ledger'],'remaining_issues':['35 exact numerical failures including preserved prior failures; no change to tolerance/math allowed','45 original API/dtype boundaries','7 original oracle errors before candidate execution','1 original checker coverage1/3 failure','1 prior original device execution error','1 unchanged fullchecker108.66GiB lower bound exceeds three96GiB cards; not numerically failed'],'changed_files_scope':str(P),'shared_registry_or_memory_writes':False}
(P/'DELIVERY.json').write_text(json.dumps(delivery,ensure_ascii=False,indent=2))
# Hash reviewable delivery/code/receipt artifacts; raw output binaries already have verified hashes in CAPTURE/REVIEW.
files=set(P.glob('*.json'))|set(P.glob('*.py'))|set(P.glob('*.patch'))
for q in P.glob('worker*'):
 if q.is_dir():
  files.update(q.glob('*.json'));files.update(q.glob('*.py'));files.update(q.glob('*.sh'));files.update(q.glob('*.patch'))
hashes={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files) if f.is_file() and f.name not in ['FILEHASHES.json','GROUP_HEARTBEAT.json','HEARTBEAT.json','GROUP_STATUS.json']}
(P/'FILEHASHES.json').write_text(json.dumps({'utc':utc,'sha256':hashes,'raw_success_output_hashes':'REVIEW_ds-v3-decode-mlp-up-bfloat16.json and REVIEW_ds-v3-prefill-mlp-up-bfloat16.json','mutable_periodic_snapshots_excluded':['GROUP_HEARTBEAT.json','HEARTBEAT.json','GROUP_STATUS.json']},indent=2))
print(json.dumps({'categories':counts,'hashed_files':len(hashes),'active_workers':hb['active'],'ready':2,'accepted_added':0},ensure_ascii=False))
