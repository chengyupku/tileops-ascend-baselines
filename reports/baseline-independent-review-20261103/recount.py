import json,pathlib,collections,subprocess,itertools,hashlib
R=pathlib.Path('/root/dyq/workspace');O=pathlib.Path(__file__).parent;P=R/'reports/baseline-v3-full-20261102/server1'
def read(p):return json.load(open(p))
def key(r):return tuple(r['key']) if 'key' in r else (r['library'],r['op'],r.get('case',r.get('case_id')))
sets={};ev={}
plan=read(P/'PLAN.json')['rows'];sets['v3_queue']={key(r) for r in plan}
a=read(O/'ARCHIVE_AUDIT.json');sets['v3_accepted']={key(r) for r in a};sets['v3_remaining']=sets['v3_queue']-sets['v3_accepted'];sets['v3_dual_role_observed']={key(r) for r in a if any(w['dual_role_tasks'] for w in r['windows'])};sets['v3_cross_task_overlap_observed']={key(r) for r in a if any(w['overlap_ns'] for w in r['windows'])}
snapshot=json.loads(subprocess.check_output(['git','-C',str(R/'.worktrees/TileOPs-site'),'show','285465c4:data/baseline-cache/snapshot.json']))
(O/'SITE_285465c4.json').write_text(json.dumps(snapshot))
sets['published']={key(r) for r in snapshot['rows']};print('libraries',collections.Counter(r['library'] for r in snapshot['rows']))
sets['torch_published']={key(r) for r in snapshot['rows'] if r['library']=='torch-eager'}
hist={};allhist=collections.defaultdict(list);source_counts={}
for f in ['docs/ascend950/results/baseline_reference_profile.jsonl','docs/ascend950/results/remeasure_20261018_v2/reference.jsonl']:
 rows=[json.loads(l) for l in open(R/f)];source_counts[f]=dict(collections.Counter(r.get('status') for r in rows))
 for n,r in enumerate(rows,1):
  if r.get('status')=='measured_reference':
   k=('torch-eager',r['op'],r['case_id']);hist[k]=r;allhist[k].append({'source':f,'line':n,'params':r['params'],'status':r['status']})
sets['torch_historical_success']=set(hist);sets['torch_unpublished']=set(hist)-sets['torch_published'];sets['original_3328_union']=sets['v3_remaining']|sets['torch_unpublished']
qops={'FP8QuantPerBlockFwdOp','INT4QuantPerGroupFwdOp','INT8QuantPerTensorFwdOp','INT8QuantPerChannelFwdOp','INT8QuantPerBlockFwdOp','SmoothQuantFwdOp'}
sets['quant6_torch']={k for k in hist if k[1] in qops}
sets['torch_historical_param_drift']={k for k,rs in allhist.items() if len({json.dumps(x['params'],sort_keys=True) for x in rs})>1}
private=read(R/'reports/ci-baseline-elementwise-20261101/BASELINE32_PRIVATE.json')['rows'];sets['private']={key(r) for r in private}
sets['elementwise_add_alternatives']={key(r) for r in private if r['key'][0] in ('tilelang','tilelang-ascend') and r['key'][1]=='AddFwdOp'}
for lib in ['tilelang-ascend','DeepSelect','sgl-kernel-npu','ops-nn']:
 sets['elementwise_'+lib]={key(r) for r in private if r['key'][0]==lib and r['case']['family']=='elementwise'}
# Composition candidates: published list is an assertion dataset, not support proof.
comp=read(P/'COMPOSITION_COVERAGE_AUDIT_20261103.json');sets['composition_prior_candidates']={key(r) for g in comp['groups'].values() for r in g['missing_inventory_candidates']}
sets['composition_claimed_max3']={key(r) for r in read(P/'COMPOSITION_MAX3_ESTIMATE_20261103.json')['retained']}
new=[]
for folder in ['reduction_full_local32_20261101','rope_local32_20261103','linear_attention_local32_20261103','norm_quant_local32_20261102','norm_server2_acceptance_20261103','reduction_server2_20261102']:
 for p in itertools.chain((R/'docs/ascend950/dev'/folder).glob('*.jsonl'), (R/'docs/ascend950/dev'/folder).glob('*/*.jsonl')):
  for n,l in enumerate(p.open(),1):
   try:r=json.loads(l)
   except ValueError:continue
   if r.get('op') and (r.get('case_id') or r.get('case')) and isinstance(r.get('eager'),dict):
    k=('torch-eager',r['op'],r.get('case_id',r.get('case')))
    if k in sets['torch_unpublished']:
     new.append({'key':k,'source':str(p.relative_to(R)),'line':n,'correct':r.get('correct'),'runtime_limit':r.get('runtime_limit'),'eager_correctness':r.get('eager_correctness'),'protocol':r['eager'].get('measurement_protocol',r.get('measurement_protocol'))})
sets['torch_unpublished_newer_eager_evidence']={tuple(r['key']) for r in new};(O/'NEWER_EAGER_EVIDENCE.json').write_text(json.dumps(new,indent=2))
if (O/'CURRENT_MANIFEST.json').exists():
 current=read(O/'CURRENT_MANIFEST.json');cm={(r['op'],r['case_id']):r for r in current['cases']}
 sets['torch_unpublished_no_current_key']={k for k in sets['torch_unpublished'] if k[1:] not in cm}
 sets['torch_unpublished_encoding_differences']={k for k in sets['torch_unpublished'] if k[1:] in cm and hist[k]['params']!=cm[k[1:]]['params']}
 sets['v3_queue_current_changed']={key(r) for r in plan if tuple(r['key'][1:]) in cm and r['case']!=cm[tuple(r['key'][1:])]}
 sets['v3_queue_current_missing']={key(r) for r in plan if tuple(r['key'][1:]) not in cm}
 sets['v3_accepted_current_changed']=sets['v3_queue_current_changed']&sets['v3_accepted']
# Normalize numeric infinity encoding, without conflating arbitrary text parameters.
 def norm(x):
  if isinstance(x,float) and x==float('inf'):return 'inf'
  if isinstance(x,float) and x==float('-inf'):return '-inf'
  if isinstance(x,dict):return {k:norm(v) for k,v in x.items()}
  if isinstance(x,list):return [norm(v) for v in x]
  return x
 sets['torch_unpublished_semantic_param_drift']={k for k in sets['torch_unpublished'] if k[1:] in cm and norm(hist[k]['params'])!=norm(cm[k[1:]]['params'])}
sets['torch_newer_v3_eligible_records']={tuple(r['key']) for r in new if str(r['protocol']).endswith('v3') and (r['eager_correctness'] or {}).get('eligible') is True}
sets['quant6_newer_eager_records']=sets['quant6_torch']&sets['torch_unpublished_newer_eager_evidence']
src=read(O/'SOURCE_CANDIDATE_RECOUNT.json')
sets['composition_source_max3']={tuple(k) for k in src['max3_source_survivors']}
sets['elementwise_slice_alternatives']={tuple(k) for k in src['slice_reconstructed'] if k[1]!='LeakyReluFwdOp'}
inv=read(R/'TileOPs/benchmarks/ascend_all_pairs/tla/inventory_snapshot.json')['cases']
sets['elementwise_uncached_activation_candidates']={('tilelang-ascend',r['op'],r['case_id']) for r in inv if r['op'] in ['SigmoidFwdOp','SiluFwdOp','SiluAndMulFwdOp'] and ('tilelang-ascend',r['op'],r['case_id']) not in sets['private']|sets['published']}
sets['v3_raw_recompute_missing_aicpu']={key(r) for r in a if any(w.get('named_uncovered_by_ffts') for w in r['windows'])}
sets['v3_complete_timing_evidence']=sets['v3_accepted']-sets['v3_raw_recompute_missing_aicpu']
qa=read(O/'QUANT_EXISTING_AUDIT.json')
sets['quant6_existing_eligible_raw_verified']={tuple(r['key']) for r in qa if r['eligible'] is True and not r['errors']}
sets['quant6_existing_ineligible']={tuple(r['key']) for r in qa if r['eligible'] is False}
sets['max3_policy_conflict_demonstrated']={key(r) for r in private if (r['key'][0]=='DeepSelect' and r['case']['family']=='elementwise') or (r['key'][0] in ['tilelang','tilelang-ascend'] and r['key'][1]=='AddFwdOp' and r['case']['tensors']['output'][1] in ['float16','bfloat16'])}
sets['current_manifest_no_historical_success']={('torch-eager',r['op'],r['case_id']) for r in current['cases']}-sets['torch_historical_success']
sets['category6_route_review_inventory']=set.union(*(sets['elementwise_'+lib] for lib in ['tilelang-ascend','DeepSelect','sgl-kernel-npu','ops-nn']))|sets['elementwise_add_alternatives']|sets['elementwise_uncached_activation_candidates']
sets['category6_alternative_candidates']=sets['elementwise_add_alternatives']|sets['elementwise_slice_alternatives']|sets['elementwise_uncached_activation_candidates']
sets['six_category_review_union']=sets['v3_queue']|sets['torch_unpublished']|sets['composition_prior_candidates']|sets['category6_route_review_inventory']
# Every set fully represented, intersections explicit.
with (O/'PAIR_SETS.jsonl').open('w') as f:
 for k in sorted(set.union(*sets.values())):
  f.write(json.dumps({'library':k[0],'op':k[1],'case':k[2],'sets':[n for n,s in sets.items() if k in s],'historical_evidence':allhist.get(k,[])},ensure_ascii=False)+'\n')
counts={n:len(s) for n,s in sets.items()};inter={a+' & '+b:len(sets[a]&sets[b]) for a,b in itertools.combinations(sets,2)}
(O/'COUNTS.json').write_text(json.dumps({'schema':1,'set_counts':counts,'intersections':inter,'source_record_counts':source_counts,'scope_note':'historical keys are not proof of current correctness; new evidence not automatically accepted'},indent=2));print(counts)
(O/'TORCH_HISTORY_DETAILS.json').write_text(json.dumps([{'key':k,'protocol':hist[k].get('measurement_protocol'),'device':hist[k]['environment'].get('device_name'),'params':hist[k]['params']} for k in sorted(sets['torch_unpublished'])],indent=2))
