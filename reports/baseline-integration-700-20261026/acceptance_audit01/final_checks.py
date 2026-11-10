import pathlib,json,hashlib,datetime,collections
O=pathlib.Path(__file__).parent;R=pathlib.Path('/root/dyq/workspace');E=json.load(open(O/'EVIDENCE_INDEX.json'));S=json.load(open(O/'SCOPE.json'));A=json.load(open(O/'RECOMMENDED_ACCEPT.json'))['rows'];facts=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert len(A)==len(S)==54
assert len({tuple(x['key']) for x in A})==54
assert {tuple(x['key']) for x in A}=={(x['baseline'],x['operator'],x['case_id']) for x in S}
for x in E:
 c=x['checker_summary'];assert c['status']=='verified';assert c['input_seed']==int.from_bytes(hashlib.sha256(f"{x['key'][1]}::{x['key'][2]}".encode()).digest()[:4],'big');assert c['inputs_preserved'] and x['inputs_metadata_equal'];assert c['verification']['checked_outputs']==c['verification']['total_outputs']>0 and c['verification']['unchecked_reason'] is None
 if x['worker']=='fla':
  for p,h in x['source'].items():
   if p.endswith('libcust_opapi.so') or p.endswith('_aclnn_ctypes.py') or p.endswith('/op_kernel/chunk_gated_delta_rule_fwd.cpp') or p.endswith('integration700_fla_native.py'):
    q=pathlib.Path(p);q=q if q.is_absolute() else R/q;actual=sha(q);assert actual==h;facts.append({'path':str(q),'expected':h,'actual':actual})
  for p,h in x['oracle'].items():
   q=pathlib.Path(p);q=q if q.is_absolute() else R/q;assert sha(q)==h
for h in ['server1','server2']:
 for suffix in ['extra_hashes','static']:
  for p,v in json.load(open(O/(h+'_'+suffix+'.json'))).items():
   if 'expected' in v:assert v['sha256']==v['expected'],(p,v)
for p,v in json.load(open(O/'DENSE_ORIGINAL_KERNEL_HASHES.json')).items():assert v['expected']==v['current_sha256']==v['original_sha256']
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
def put(n,j):
 p=O/n;q=p.with_suffix('.tmp');q.write_text(json.dumps(j,indent=2,ensure_ascii=False));q.replace(p)
put('FINAL_CHECKS.json',{'time':now,'scope_exact_54_unique':True,'all_54_full_checker_and_seed_input_preservation':True,'remote_selected_pins_match':True,'fla_primary_runtime_checks':facts,'NPU_commands':0,'device_imports':0,'numerical_reruns':0,'external_writes':0})
# Link topk import-only proof to the final rows and family artifact.
for row in A:
 if row['key'][1]=='FusedTopKFwdOp':row['source_provenance']['formal_import_relocation_proof']=str(O/'TOPK_IMPORT_RELOCATION_PROOF.json')
put('RECOMMENDED_ACCEPT.json',{'at':now,'rows':A})
put('DENSE_RECOMMENDED_ACCEPT.json',{'rows':[a for a in A if a['key'][1] in ['FusedTopKFwdOp','LongRoPEFwdOp','YaRNFwdOp','GeluTanhAndMulFwdOp']]})
summary={'worker':'e0ac5d5a47ab4783','started_at':json.load(open(O/'STARTED.json'))['time'],'finished_at':now,'scope_count':54,'recommend_accept':sum(r['accept_recommended'] for r in A),'needs_evidence':sum(not r['accept_recommended'] for r in A),'reject':0,'families':{},'pending_keys':[r['key'] for r in A if not r['accept_recommended']],'issues':['YaRN live formal module differs from successful checker; missing exact old source branch-equivalence and original-kernel launch attribution. See ALERTS.json. Guard exit1 belongs to later LongRoPE failure and is preserved as 1.'],'evidence_entrypoints':['RECOMMENDED_ACCEPT.json','EVIDENCE_INDEX.json','NATIVE_LOG_AUDIT.json','FINAL_CHECKS.json','ALERTS.json'],'major_waste_observed':['Scope RESULT often points to synthetic per-case shared_guard receipt; actual parent guard must be joined using explicit batch argv, keys, ticket/PID and timestamps. Unordered glob mapping is unsafe.','Shared-process logs remain under first case: checking only later empty files would create needless reruns. This audit matched 12 attention cases to native compile and exact-time device launches.','Unfrozen adapter edits after checker complicate registration acceptance: TopK was resolved by exact reconstructed tested digest and import-only diff to byte-identical GraphCall; YaRN remains pending.','Large source snapshots include unrelated installed package/SDK files; only key API/kernel/runtime hashes plus manifest identity were read, no full SDK hash sweep.'],'constraints':{'NPU_forbidden_observed':True,'numerical_reruns':0,'memory_writes':0,'ACCEPTED_json_modified':False,'writes_only_audit_directory':True},'notes':['Recommendations only; final accounting belongs to parent. No CLI completion flag was used as acceptance proof.','TopK raw index max_abs_err is large due to ordering; unchanged canonical custom verifier checks expert sets, uniqueness and paired weights, not positional index equality.','TLA Neg/Clamp row slices are independent original API calls with untouched tile/pass configs; concatenation is output packing, not new numerical decomposition.']}
for w in dict.fromkeys(r['worker'] for r in S):
 keys={(r['baseline'],r['operator'],r['case_id']) for r in S if r['worker']==w};rs=[r for r in A if tuple(r['key']) in keys];summary['families'][w]={'scope':len(rs),'recommend_accept':sum(r['accept_recommended'] for r in rs),'needs_evidence':sum(not r['accept_recommended'] for r in rs),'reject':0}
put('AUDIT_SUMMARY.json',summary);put('HEARTBEAT.json',{'time':now,'status':'completed','reviewed':54,'recommend_accept':summary['recommend_accept'],'needs_evidence':summary['needs_evidence']});print(json.dumps({k:summary[k] for k in ['scope_count','recommend_accept','needs_evidence','reject','families']},ensure_ascii=False))
