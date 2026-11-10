import json,pathlib,statistics,collections,datetime
O=pathlib.Path(__file__).parent;R=pathlib.Path('/root/dyq/workspace');a=json.load(open(O/'ARCHIVE_AUDIT.json'));q=json.load(open(O/'QUANT_EXISTING_AUDIT.json'));c=json.load(open(O/'COUNTS.json'));rows=[json.loads(l) for l in open(O/'PAIR_SETS.jsonl')]
assert len(a)==621 and not any(r['errors'] for r in a)
assert sum(len(r['windows']) for r in a)==9315
assert all(len({w['trace_sha'] for w in r['windows']})==15 for r in a)
assert all(w['devices']==1 and not w['stars_errors'] for r in a for w in r['windows'])
missing=[r for r in a if any(w['named_uncovered_by_ffts'] for w in r['windows'])];assert len(missing)==1
assert len(q)==46 and sum(r['eligible'] is True for r in q)==39 and not any(r['errors'] for r in q)
for r in q:
 src=[json.loads(l) for l in open(R/r['source'])];x=next(x for x in src if x['case']==r['key'][2]);assert x['inputs_preserved_after_timing'];v=x['verification'];assert v['checked_outputs']==v['total_outputs']>0
 if r['eligible']:
  v=x['eager_correctness']['verification'];assert v['checked_outputs']==v['total_outputs']>0
actual=collections.Counter(n for r in rows for n in r['sets']);assert all(actual[k]==v for k,v in c['set_counts'].items())
r=missing[0];samples=[w['task_span_sum_ns']/1e6+sum(t['duration_us'] for t in w['missing_named_stars'])/1000 for w in r['windows']]
fix={'key':r['key'],'task_id':r['task_id'],'source_archive':r['archive'],'metric':'sum of distinct FFTS task spans plus missing named AICPU durations; no observed cross-task FFTS overlap','old_latency_ms':statistics.mean(w['task_span_sum_ns']/1e6 for w in r['windows']),'corrected_latency_samples_ms':samples,'corrected_latency_ms':statistics.mean(samples),'new_npu_measurement_required':False,'source_evidence':'ARCHIVE_AUDIT.json windows.missing_named_stars','not_written_back_to_original':True}
(O/'W4A16_RECOMPUTE.json').write_text(json.dumps(fix,indent=2))
findings=[{'id':'F01','severity':'critical','status':'confirmed','pairs':1,'set':'v3_raw_recompute_missing_aicpu','action':'CPU raw recompute then export; do not rerun NPU'}, {'id':'F02','severity':'high','status':'confirmed inventory/interpretation error','pairs':385,'set':'torch_unpublished_newer_eager_evidence','action':'audit existing evidence before scheduling retest'}, {'id':'F03a','severity':'high','status':'confirmed usable stored eager evidence','pairs':39,'set':'quant6_existing_eligible_raw_verified','action':'export admission'}, {'id':'F03b','severity':'high','status':'confirmed ineligible eager','pairs':7,'set':'quant6_existing_ineligible','action':'keep excluded; investigate original backend/checker mismatch only when separately authorized'}, {'id':'F04','severity':'high','status':'confirmed conditional policy inconsistency','pairs_lower_bound':26,'set':'max3_policy_conflict_demonstrated','action':'apply max3 scope consistently; do not silently delete historically authorized measurements'}, {'id':'F05','severity':'medium','status':'parser capability defect, actual impact unconfirmed beyond F01','pairs':None,'evidence':'PARSER_CONTROLS.json'}, {'id':'F06','severity':'medium','status':'confirmed omitted no-history universe','pairs':400,'set':'current_manifest_no_historical_success','action':'separate failure/device fault/no-record inventory; do not assume support'}]
(O/'FINDINGS.json').write_text(json.dumps(findings,ensure_ascii=False,indent=2))
(O/'VALIDATION.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive_pairs':621,'archive_windows':9315,'complete_named_coverage_pairs':620,'missing_aicpu_pairs':1,'unique_trace_sha_each_pair':15,'quant_pairs':46,'quant_windows':1380,'quant_eligible':39,'quant_ineligible':7,'quant_guard_and_raw_errors':0,'set_recount_matches':True,'npu_calls':0,'remote_calls':0,'source_mutations':0},indent=2))
print('Validated',fix['corrected_latency_ms'])
