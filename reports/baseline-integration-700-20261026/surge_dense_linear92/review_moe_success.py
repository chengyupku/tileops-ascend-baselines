import pathlib,json,hashlib,datetime,collections
P=pathlib.Path(__file__).resolve().parent;W=pathlib.Path('/root/dyq/workspace');worker=P/'worker04moe'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
reviews=[]
for q in worker.glob('ds-v3-*'):
 f=q/'result/representative.json'
 if not f.exists():continue
 c=json.loads(f.read_text())
 if c.get('status')!='verified':continue
 d=q/'result';g=json.loads((q/'GUARD_RESULT.json').read_text());cap=json.loads((d/'CAPTURE.json').read_text());ex=json.loads((d/'EXECUTION.json').read_text());n=json.loads((d/'NATIVE_CALLS.json').read_text());entries=[x for x in n if x.get('event')=='enter'];returns=[x for x in n if x.get('event')=='return'];native_paths={x['path']:x['sha256'] for x in entries};source_checks={s:sha(W/s)==h for s,h in c['source_sha256'].items()};native_checks={s:sha(pathlib.Path(s))==h for s,h in native_paths.items()};outputs={}
 for name,v in cap.items():
  if v.get('stored'):
   out=d/(name+'.bin');outputs[name]={'bytes':out.stat().st_size,'expected_bytes':v['bytes'],'sha256_match':sha(out)==v['sha256']}
 checks={'exact_baseline':c['library']=='ops-transformer','case_directory_matches':c['case_id']==q.name,'original_checker_verified':c['status']=='verified','all_outputs_checked':c['verification']['checked_outputs']==c['verification']['total_outputs']==1,'inputs_preserved':c['inputs_preserved'] is True,'guard_success':g['exit_code']==0,'guard_finished_clean':g['metadata_after'].get('phase')=='finished' and not g['metadata_after'].get('conflict_detected') and not g['metadata_after'].get('visibility_lost'),'frozen_unchanged':ex['frozen_unchanged'],'provider_oracle_unchanged':not c['provider_source_changed'] and not c['oracle_source_changed'],'sources_unchanged':all(source_checks.values()),'native_library_unchanged':all(native_checks.values()),'native_map_identity':all(x['map_inode']==x['stat_inode'] for x in entries),'all_native_returns_zero':all(x['return_code']==0 for x in returns),'native_pair_count':len(entries)==len(returns)==512,'full_output_capture_hashes':len(outputs)==2 and all(v['sha256_match'] and v['bytes']==v['expected_bytes'] for v in outputs.values())}
 review={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'key':[c['library'],c['op'],c['case_id']],'evidence_review_passed':all(checks.values()),'checks':checks,'source_hash_checks':source_checks,'native_hash_checks':native_checks,'full_outputs':outputs,'seed':c['input_seed'],'native_functions':dict(collections.Counter(x['function'] for x in entries)),'checker':str(f),'guard':str(q/'GUARD_RESULT.json'),'native':str(d/'NATIVE_CALLS.json'),'capture':str(d/'CAPTURE.json'),'execution':str(d/'EXECUTION.json'),'prior_original_source_build_proof':str(P.parent/'bf16_moe_down_full3/PARENT_REVIEW.json'),'review_scope':'Reuse unchanged already-reviewed original build/source provenance; verify actual native mapping/source diff/all outputs/input preservation/guard only. No NPU rerun.','accepted_added':0}
 (P/('REVIEW_'+q.name+'.json')).write_text(json.dumps(review,ensure_ascii=False,indent=2));reviews.append(review)
print(json.dumps([{'key':r['key'],'passed':r['evidence_review_passed'],'failed_checks':[k for k,v in r['checks'].items() if not v]} for r in reviews],ensure_ascii=False))
