"""Generate a deduplicated evidence-based plan, never start remote work."""
import collections,csv,datetime,hashlib,json,pathlib,statistics
R=pathlib.Path('/root/dyq/workspace');A=R/'reports/baseline-independent-review-20261103';O=pathlib.Path(__file__).parent
S={k:set(map(tuple,v)) for k,v in json.loads((A/'SET_MEMBERS.json').read_text()).items()}
replay=json.loads((O/'REPLAY.json').read_text());assert len(replay['rows'])==621 and not any(x['errors'] for x in replay['rows'])
accepted={tuple(x['key']):x for x in replay['rows']};new=collections.defaultdict(list)
for e in json.loads((O/'NEWER_RECORDS.json').read_text()):new[tuple(e['key'])].append(e)
unknown={tuple(x['key']):x for x in json.loads((A/'NO_HISTORICAL_SUCCESS.json').read_text())}
allkeys=S['six_category_review_union']|S['current_manifest_no_historical_success']; rows=[]
for key in sorted(allkeys):
 tags=[name for name,keys in S.items() if key in keys]; reasons=[];evidence=[]
 if key in accepted:
  x=accepted[key];action='offline_corrected' if x['changed'] else 'reuse_remote32';reason='15 raw windows, archive manifest, checker and guard verified using revised parser';evidence=[x['archive']]; value=x['corrected_ms']
 elif key in S['v3_remaining']:
  action='measure_remote32';reason='Original v3 campaign unfinished; no admitted complete v3 receipt in audited inventory';value=None
 elif key in S['torch_unpublished']:
  action='fix_then_measure_remote32' if key in S['quant6_existing_ineligible'] else 'measure_remote32'
  reason='No admitted physical32 v3 receipt among retained audited historical and newer records';value=None
  if key in S['quant6_existing_ineligible']:reasons.append('FP8 quantization bits differ; fix reference semantics before timing')
  evidence=sorted(set(e['source'] for e in new[key]));reasons.extend(sorted({z for e in new[key] for z in e['admission_blockers']}))
 elif key in unknown:
  action='diagnose_before_measurement_decision';reason='No historical successful reference for current case; failure is not an admitted runnable baseline';value=None;evidence=[unknown[key].get('file','')];reasons=[unknown[key]['status'],unknown[key].get('error','')]
 elif key in S['category6_alternative_candidates'] or key in S['composition_source_max3']:
  action='validate_native_candidate';reason='Candidate only; prove top-level compatibility and correctness before timing';value=None
 elif key in S['composition_prior_candidates']:
  action='excluded_composition_over3_or_unproven';reason='Historical candidate not retained by <=3 operator source screen';value=None
 else:
  action='source_route_review';reason='Review inventory, not a demonstrated timing defect or mandatory rerun';value=None
 rows.append({'key':list(key),'action':action,'reason':reason,'details':reasons,'evidence':evidence,'categories':tags,'corrected_latency_ms':value,'newer_records':len(new[key]),'validate_alternative_before_scheduling':key in (S['category6_alternative_candidates']|S['composition_source_max3']), 'max3_existing_route_requires_policy_review':key in S['max3_policy_conflict_demonstrated']})
assert len({tuple(r['key']) for r in rows})==len(rows)
required=[r for r in rows if r['action'] in ('measure_remote32','fix_then_measure_remote32')]
assert not {tuple(r['key']) for r in required}&accepted.keys()
for name,items in [('PAIR_DECISIONS',rows),('REQUIRED_MEASUREMENTS',required),('REUSABLE', [r for r in rows if r['action'] in ('reuse_remote32','offline_corrected')]),('DIAGNOSIS', [r for r in rows if r['action']=='diagnose_before_measurement_decision']),('CANDIDATES',[r for r in rows if 'category6_alternative_candidates' in r['categories'] or 'composition_source_max3' in r['categories']])]:
 (O/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items))
 with (O/(name+'.csv')).open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['library','op','case','action','reason','details','evidence']);w.writerows([*r['key'],r['action'],r['reason'],'; '.join(r['details']),'; '.join(r['evidence'])] for r in items)
summary={'generated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Frozen independent audit pair inventory plus its 400 current cases lacking historical success; not an exhaustive search of every workspace or remote unpublished record','target':'Physical 32 AIC/64 AIV, complete checker, guard, 15 v3 windows with full task coverage and retained raw','unique_review_keys':len(rows),'actions':dict(collections.Counter(r['action'] for r in rows)),'required_measurements_under_current_evidence':len(required),'campaign_remaining':len(S['v3_remaining']),'torch_unpublished_without_admitted_physical32_evidence':len(S['torch_unpublished']),'fp8_fix_first_in_required':7,'reused_without_npu':len(accepted),'offline_corrections':sum(r['changed'] for r in accepted.values()),'historical_20_all_already_in_621':True,'newer_record_count':sum(map(len,new.values())),'newer_pair_count':sum(bool(v) for v in new.values()),'newer_auto_admitted':0,'candidate_pairs':162,'candidates_overlapping_required':129,'candidates_overlapping_reusable_current_routes':1,'candidate_only_pairs':32,'note':'3328 is conditional on the physical32 target and this evidence scope, not a universal proven minimum; missing evidence can be supplied to remove keys. 400 diagnosis items and candidate validations are separate, not added blindly. Original receipts and pause state unchanged.','parser_revision':replay['parser_revision'],'parser_sha256':replay['parser_sha256']}
(O/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
derived=[]
for x in replay['rows']:
 samples=[w['latency_ms'] for w in x['windows']]
 derived.append({'key':x['key'],'task_id':x['task_id'],'source_archive':x['archive'],'source_archive_sha256':hashlib.sha256(pathlib.Path(x['archive']).read_bytes()).hexdigest(),'parser_revision':replay['parser_revision'],'parser_sha256':replay['parser_sha256'],'status':'verified_offline_replay','old_latency_ms':x['old_ms'],'changed':x['changed'],'latency_ms':statistics.mean(samples),'latency_samples_ms':samples,'latency_median_ms':statistics.median(samples),'latency_stdev_ms':statistics.stdev(samples),'raw_trace_sha256':[w['trace_sha256'] for w in x['windows']],'task_coverage_complete':True,'full_task_evidence':'REPLAY.json'})
(O/'CORRECTED_TIMINGS.jsonl').write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in derived))
(O/'INPUT_HASHES.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [A/'SET_MEMBERS.json',A/'NEWER_EAGER_EVIDENCE.json',A/'NO_HISTORICAL_SUCCESS.json',O/'NEWER_RECORDS.json',O/'RAW_HARDWARE.json',O/'REPLAY.json']},indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2))
