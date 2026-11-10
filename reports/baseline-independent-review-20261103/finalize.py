import json,pathlib,collections,hashlib,csv,datetime,itertools,statistics
R=pathlib.Path('/root/dyq/workspace');O=pathlib.Path(__file__).parent
rows=[json.loads(l) for l in (O/'PAIR_SETS.jsonl').open()];sets=collections.defaultdict(set)
for r in rows:
 for name in r['sets']:sets[name].add((r['library'],r['op'],r['case']))
c=json.load(open(O/'COUNTS.json'));c['as_of_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();c['pair_key']=['library','op','case'];c['categories']=[
 {'id':1,'original_claim':2814,'verified_inventory':2814,'confirmed_old_v2_bad_pair_count':None,'new_trace_demonstrated_double_count_if_v2_used':406,'classification':'2814 both-cycle screening keys; neither all confirmed invalid nor all beyond repair'},
 {'id':2,'original_claim':{'accepted':621,'remaining':2193},'ledger_accepted':621,'independently_complete_timing':620,'raw_recompute':1,'required_new_npu_measurement':None,'campaign_remaining':2193,'reexport_after_recompute':621},
 {'id':3,'historical_success':1662,'published_snapshot':527,'historical_not_published':1135,'operators':len({k[1] for k in sets['torch_unpublished']}),'newer_32core_eager_records':385,'newer_v3_eligible_records':202,'current_manifest_no_historical_success':len(sets['current_manifest_no_historical_success']),'mandatory_retest_count':None,'original_3328_arithmetic_union':3328,'reason_not_total_retest':'existing newer records, raw-recompute work, reference failures and no-history cases require distinct admission'},
 {'id':4,'historical_quant6':46,'contained_in_1135':46,'newer32_v3_raw_and_guard_checked':46,'eligible_for_export_admission':39,'ineligible_FP8_block':7,'confirmed_new_native_library_support':0,'universal_no_native_support_proven':False},
 {'id':5,'historical_candidates_reconstructed':260,'max3_source_survivors':8,'verified_new_pairs':0,'survivors_are_exhaustive_support':False,'existing_route_max3_conflict_lower_bound':26},
 {'id':6,'add_candidates':24,'slice_candidates':118,'uncached_activation_candidates':12,'candidate_union':154,'four_library_cached_elementwise_inventory':799,'review_inventory_including_tilelang_add_and_uncached':823,'proven_faster_alternatives':0}]
c['decision_counts']={'complete_621_reexport_candidates':620,'cpu_raw_recompute_missing_aicpu':1,'quant39_export_admission_candidates':39,'quant7_ineligible_reference':7,'additional_new_v3_eligible_records_requiring_receipt_audit':len(sets['torch_newer_v3_eligible_records']-sets['quant6_existing_eligible_raw_verified']),'mandatory_retest_total':None,'alternative_device_validation_candidates':154,'composition_device_validation_candidates':8}
c['limitations']=['No NPU, remote task, live-site request, service action or source modification was performed.','385/202 are frozen located-record inventories, not accepted global minima or exhaustive newer-source coverage.','620 timing coverage acceptance is against retained named-task traces, not proof of absence of invisible untracked transfers or fresh universal correctness.','The 26 max3 conflicts apply the requested new counting rule to identified adapters; historical authorization is not retroactively judged.']
# Failed/no-success inputs are a separate universe from the historical success gap.
latest={}
for f in ['docs/ascend950/results/baseline_reference_profile.jsonl','docs/ascend950/results/remeasure_20261018_v2/reference.jsonl']:
 for i,l in enumerate(open(R/f),1):
  r=json.loads(l)
  if r.get('op') and r.get('case_id'):latest[('torch-eager',r['op'],r['case_id'])]={'status':r.get('status'),'file':f,'line':i,'error':r.get('error')}
missing=[{'key':k,**latest.get(k,{'status':'no_record'})} for k in sorted(sets['current_manifest_no_historical_success'])]
(O/'NO_HISTORICAL_SUCCESS.json').write_text(json.dumps(missing,indent=2));c['current_no_historical_success_statuses']=dict(collections.Counter(r['status'] for r in missing))
(O/'COUNTS.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n')
with (O/'PAIR_SETS.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['library','op','case','sets'])
 for r in rows:w.writerow([r['library'],r['op'],r['case'],';'.join(r['sets'])])
(O/'SET_MEMBERS.json').write_text(json.dumps({n:sorted(s) for n,s in sets.items()},indent=2))
# Compact source evidence hashes: these prove the files reviewed, not remote execution by themselves.
paths=[R/'reports/baseline-v3-full-20261102/server1'/n for n in ['PLAN.json','PROGRESS.json','POLICY.json','PAUSE.json','controller.py','bootstrap.py','v3/kernel_task_profile.py','v3/measure_baselines.py']]
paths += [R/p for p in ['docs/ascend950/results/baseline_reference_profile.jsonl','docs/ascend950/results/remeasure_20261018_v2/reference.jsonl','reports/ci-baseline-elementwise-20261101/BASELINE32_PRIVATE.json','reports/ci-baseline-elementwise-20261101/export_verified_results.py','TileOPs/benchmarks/ascend_all_pairs/tla/adapters.py','TileOPs/benchmarks/ascend_all_pairs/tla/plans.py','TileOPs/benchmarks/ascend_all_pairs/deepselect/adapters.py','TileOPs/benchmarks/ascend_all_pairs/sgl/additional.py','TileOPs/benchmarks/ascend_all_pairs/sgl/comparison.py','TileOPs/benchmarks/ascend_all_pairs/sgl/finite_truth.py','TileOPs/benchmarks/ascend_all_pairs/ops_nn/elementwise.py','TileOPs/benchmarks/ascend_all_pairs/ops_nn/truth.py','TileOPs/benchmarks/ascend_all_pairs/ops_nn/normalization.py','TileOPs/benchmarks/ascend_all_pairs/ops_math/composition.py','TileOPs/benchmarks/ascend_all_pairs/tilelang/composed.py','TileOPs/benchmarks/ascend_tilelang_more.py','TileOPs/workloads/quantization/quantize.py','tools/measure_baselines.py','tools/npu_guard.py']]
paths += [R/p for p in {r['source'] for r in json.load(open(O/'NEWER_EAGER_EVIDENCE.json'))}]
(O/'EVIDENCE_HASHES.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
print(c['set_counts']);print(c['current_no_historical_success_statuses'])
