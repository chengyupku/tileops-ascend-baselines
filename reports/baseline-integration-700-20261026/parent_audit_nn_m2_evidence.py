import pathlib,json,hashlib,datetime,re
W=pathlib.Path('/root/dyq/workspace');D=W/'reports/baseline-integration-700-20261026';P=D/'nn_m2_source_audit1';R=D/'nn_recovery02/matrix_05_01'
read=lambda p:json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
packet=read(P/'PACKET_HASHES.json');assert all(sha(P/k)==v for k,v in packet.items())
for x in read(P/'EVIDENCE_HASHES.json'):assert sha(x['path'])==x['sha256'],x['path']
c=read(R/'checker.json');rec=read(P/'SOURCE_RACE_RECONCILIATION.json');assert c['status']=='source_changed' and c['inputs_preserved'];assert c['verification']['checked_outputs']==c['verification']['total_outputs']==1 and c['verification']['unchecked_reason'] is None
assert not c.get('error') and not c.get('cleanup_error') and not c.get('synchronize_error') and not c['oracle_source_changed'];assert c['input_seed']==3206609422
assert set(c['provider_source_changed'])==set(rec['changed_paths']) if isinstance(c['provider_source_changed'],list) else c['provider_source_changed']
for field in ['source_sha256','oracle_source_sha256']:
 for p,h in c[field].items():assert sha(W/p)==h,p
assert sha(W/'TileOPs/scripts/ascend_pair_check.py')==c['verifier_sha256']
for x in rec['original_kernel_pairs']:assert sha(x['installed'])==sha(x['original'])==x['sha256']
b=pathlib.Path(rec['before']['path']).read_text();a=pathlib.Path(rec['after']['path']).read_text();assert a.startswith(b);suffix=a[len(b):];assert 'GemmFwdOp' not in suffix and 'GemmW4A16FwdOp' in suffix
assert c['provider_source_sha256']['TileOPs/benchmarks/ascend_all_pairs/ops_nn/__init__.py']==rec['before']['sha256'];assert c['provider_source_sha256_after']['TileOPs/benchmarks/ascend_all_pairs/ops_nn/__init__.py']==rec['after']['sha256']
assert sha(R/'adapter.py')==sha(P/'adapter_037f902f6779aa1d84dad1d5d04777e6754cec5e9b3d653b7339ac5f0dd200d7.py')=='037f902f6779aa1d84dad1d5d04777e6754cec5e9b3d653b7339ac5f0dd200d7'
events=[json.loads(s) for s in (R/'native.jsonl').read_text().splitlines()];raw={e['file']:pathlib.Path(e['file']).read_text().splitlines() for e in events}
for e in events:assert raw[e['file']][e['line']-1]==e['text']
n=read(P/'NATIVE_EVIDENCE.json');kernel='te_matmulv3_e1f1ffb460145f29bb9ab7c5fe0583e81f7f24a5e73338b264f6223c4b25db2a';txt='\n'.join(e['text'] for e in events)
assert kernel in txt and n['logged_kernel_sha256'] in txt and 'classicpilot_nn_impl.dynamic.mat_mul_v3' in txt and 'result[success]' in txt and 'StreamLaunchKernelV1' in txt
for e in n['events']:
 tm=re.search(r'(2026-10-26-\d\d:\d\d:\d\d)\.(\d+)\.(\d+)',e['text']);assert tm;ts=tm[1].replace('2026-10-26-','2026-10-26T')+'.'+tm[2]+tm[3]+'+00:00'
 assert datetime.datetime.fromisoformat(c['started_at'])<=datetime.datetime.fromisoformat(ts)<=datetime.datetime.fromisoformat(c['finished_at'])
assert read(D/'nn_recovery02/matrix_batch01/GUARD_RESULT.json')['returncode']==0
checker=(W/'TileOPs/scripts/ascend_pair_check.py').read_text();fin=checker.index('    finally:');assert checker.index('torch.npu.synchronize()',fin)<checker.index('args.output.write_text',fin);assert checker.index('candidate.cleanup()',fin)<checker.index('args.output.write_text',fin)
r={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'key':['ops-nn','GemmFwdOp','ds-v3-m2-qkv-a-bfloat16'],'status':'eligible_pending_formal_registration_after_active_nn_checks','accepted_added':0,'packet_hashes_verified':len(packet),'historical_dependency_hashes_verified':len(c['source_sha256']),'oracle_hashes_verified':len(c['oracle_source_sha256']),'native_raw_lines_verified':len(events),'source_race_reconciled':True,'historical_checker_status':'source_changed','historical_child_rc':1,'actual_batch_guard_rc':0,'original_boundary':'ExactsameimmutableBF16-to-FP32 completeMatMulV3 binding aspreviouslyacceptedNN7. Full4224elements,inputpreservation,canonicalseed/tolerance,sync/cleanup retained. UnrelatedW4registryappend reconciled fromhash-pinned before/after.','native_evidence_decision':'The originalrawcompile-success log namesclassicpilot_nn source/module, generatedkernel name andELFSHA256; samePID launches samekernel withinchecker/guardinterval. All2170source/runtime hashes match. This establishes originalnativeattribution under the already-used NN7 acceptance criteria. Physical JITELF retention is an additional audit capability, not an independently stated required gate; do not rerun a successful numericalcase solely to regenerate it.','qualification':'HistoricalgeneratedJITELFbytes notretained; loggedSHA256not independentlyrehashed. Keepthislimitation and source_changed/child1/batch0 explicitly. Formalregistry stillpending and no acceptanceadded.','parent_did_not_apply_patch':True,'new_npu_runs':0,'registration_wait_for':'nn_dense_resume8 completion toavoidnewprovider race','evidence':['RECOMMENDED_ACCEPT.json','SOURCE_RACE_RECONCILIATION.json','NATIVE_EVIDENCE.json','CPU_PROOF.json','REGISTRATION.patch']}
(P/'PARENT_REVIEW.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(r)
