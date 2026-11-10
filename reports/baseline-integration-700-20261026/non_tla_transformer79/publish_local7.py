import pathlib,json,hashlib,datetime,fcntl
G=pathlib.Path(__file__).resolve().parent;Q=G/'fp8_local_evidence7';v=json.loads((Q/'REVIEW.json').read_text());out=[];checks=[]
for r in v['rows']:
 c=json.loads(pathlib.Path(r['checker']['path']).read_text());guard=json.loads(pathlib.Path(r['guard']['path']).read_text());n=json.loads(pathlib.Path(r['native']['path']).read_text())
 assert [c['library'],c['op'],c['case_id']]==r['key']
 assert c['status']=='verified' and c['verification']['checked_outputs']==c['verification']['total_outputs']==1
 assert c['inputs_preserved'] and not c['oracle_source_changed'] and not c['provider_source_changed']
 assert guard['returncode']==0 and guard['guard_metadata']['workload_completed'] and not guard['guard_metadata']['conflict_detected'] and not guard['guard_metadata']['visibility_lost']
 assert len(n)==1 and n[0]['native_calls']==1 and n[0]['synchronized'] and n[0]['api']=='aclnnGroupedMatmulV5'
 for label in ['checker','guard','native','loaded_maps']:
  p=pathlib.Path(r[label]['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r[label]['sha256']
 checks.append({'key':r['key'],'complete_checker':True,'original_oracle_unchanged':True,'single_original_native_call':True,'guard_clean_completed':True,'registered_by_external_local_owner':True})
 out.append({'key':r['key'],'status':'verified_ready_for_parent_serial_publication','ready_to_publish':True,'evidence_review_passed':True,'registered':True,'accepted_added':0,'review':str(G/'REVIEW_LOCAL7.json'),'worker_review':str(Q/'REVIEW.json'),'checker':r['checker']['path'],'guard':r['guard']['path'],'native':r['native']['path'],'execution':r['guard_running']['path'],'loaded_maps':r['loaded_maps']['path'],'registration_pins':str(pathlib.Path(r['runtime_launcher']).parent/'FP8_RUNTIME_PREFLIGHT.json'),'registration_evidence':r['registration_evidence'],'runtime_launcher':r['runtime_launcher'],'input_seed':c['input_seed'],'verification':c['verification'],'source_identity':v['identity'],'adaptation':'Single complete original GEMM; zero-row padding, operand exchange, and output selection. All copies and padding in run. 128x output storage; no performance claim.','reused_external_success':True,'device_rerun_required':False,'acceptance_owner':'acceptance_root'})
review={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reviewed_count':7,'accepted_added':0,'checks':checks,'semantic_decision':'Approve source/math-preserving one-complete-call padding mapping for these seven exactkeys; no multi-call composition. Existing local registration is separately authorized and was not written by this root.','evidence_reuse':str(Q/'REVIEW.json'),'identity_limit':'Original compiler absolute recipe and version plus original source/copy association preserved; no compiler executable hash was recorded. Do not misrepresent this as a hash receipt; acceptance root decides under existing identity policy, without inventing success rerun.'}
(G/'REVIEW_LOCAL7.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
with (G/'CHILD_DISPATCH.lock').open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX);p=G/'PUBLISH_QUEUE.json';old=json.loads(p.read_text()) if p.exists() else {'entries':[]};keys={tuple(x['key']) for x in out};old['entries']=[x for x in old['entries'] if tuple(x['key']) not in keys]+out;old['at']=review['at'];p.write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n')
print('7 reviewed exact successes queued to acceptance_root; accepted delta remains zero')
