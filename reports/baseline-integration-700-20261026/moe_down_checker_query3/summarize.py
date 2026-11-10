import json,hashlib,time,re,math
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent
plan=json.loads((H/'PLAN.json').read_text()); results={}; checks=[]; actual=[]
for s in plan:
 d=H/(s['id']+('_hugefirst' if s['id']=='p00_B' else ''));r=json.loads((d/'result.json').read_text());g=json.loads((d/'guard_receipt.json').read_text());assert g['rc']==0 and r['status']=='queried';assert all(c['rc']==0 for c in r['cleanup']);assert len([c for c in r['cleanup'] if c['name']=='aclDestroyAclOpExecutor'])==len(r['queries']);assert all(q['rc']==0 and q['executor'] for q in r['queries'])
 assert r['admission']['hbm']['free']>=s['payload_bytes']+32*2**30;assert r['admission']['host']['effective_available_host_bytes']>=32*2**30
 descriptors=[x for x in r['events'] if x['name']=='descriptor'];assert len(descriptors)==4 and len({x['device_ptr'] for x in descriptors})==4
 logs=list((d/'runtime/ASCEND_PROCESS_LOG_PATH').rglob('*.log'));records=[]
 for log in logs:
  for i,l in enumerate(log.read_text(errors='replace').splitlines(),1):
   if re.search(r'OpName:\[aclnn(IsClose|All)_\d+\] Entering function params',l):records.append({'file':str(log),'line':i,'text':l})
 for api in ['IsClose']+(['All'] if s['query_all'] else []):
  rec=[x for x in records if 'OpName:[aclnn'+api+'_' in x['text']];assert len(rec)>=2
  selftext=next(x['text'] for x in rec if 'params: self:' in x['text']);outtext=next(x['text'] for x in rec if 'params: out:' in x['text'])
  for field,value in [('view_shape',s['shape']),('view_strides',s['strides']),('storage_shape',s['storage_shape'])]: assert f'{field} = {value}' in selftext
  assert 'view_format = '+s['format'] in selftext and 'storage_format = '+s['format'] in selftext
  assert ('view_shape = []' if api=='All' else 'view_shape = '+str(s['shape'])) in outtext
 actual.extend(records);results[s['id']]=r
 checks.append({'id':s['id'],'guard_exit':g['rc'],'elapsed_seconds':g['end']-g['start'],'actual_descriptor_trace_verified':True,'executor_destructor_all_rc0':True,'result':str(d/'result.json'),'queries':r['queries'],'cleanup_seconds':r['cleanup_seconds']})
assert sum(len(r['queries']) for r in results.values())==13
(H/'ACTUAL_TRACE_METADATA.json').write_text(json.dumps(actual,indent=2))
old=json.loads((P/'moe_fullchecker_scratch5/PER_KEY_BOUNDS.json').read_text())['keys'];scope={x[0] for s in plan for x in s['uses']};keys=[]
for row in old:
 if row['case_id'] not in scope:continue
 ws={}
 for s in plan:
  for key,operand in s['uses']:
   if key!=row['case_id']:continue
   r=results[s['id']];close=r['queries'][0]['workspace_bytes'];ar=results[s['all_reference'] or s['id']];allws=next(q['workspace_bytes'] for q in ar['queries'] if q['api']=='aclnnAllGetWorkspaceSize');ws[operand]={'probe':s['id'],'IsClose':close,'All':allws,'max_sequential':max(close,allws)}
 # Deliberately conservative sum of explicit maximum and workspace maximum; no extra hypothetical full BOOL internal output.
 ip=row['input_preservation_success'];dr=row['defined_rows_success'];base=row['inherited_live_base_bytes'];v5=256*512
 success=max(ip['explicit_python_temp_max_bytes']+max(ws[x]['max_sequential'] for x in ['A','B','metadata']),dr['explicit_value_temp_max_bytes']+ws['checker']['max_sequential'])
 keys.append({'case_id':row['case_id'],'inherited_base_bytes':base,'api_workspace_bytes':ws,'source_conditional_input_preservation_explicit_peak_bytes':ip['explicit_python_temp_max_bytes'],'source_conditional_defined_rows_explicit_peak_bytes':dr['explicit_value_temp_max_bytes'],'source_conditional_success_base_plus_explicit_plus_observed_workspace_bytes':base+success,'conditional_all_256_V5_workspace_bytes_reused':v5,'source_conditional_success_with_V5_bytes':base+success+v5,'source_conditional_B_failure_explicit_peak_bytes':row['input_B_failure']['explicit_large_tensor_temp_peak_bytes'],'source_conditional_B_failure_base_plus_explicit_bytes':row['input_B_failure']['base_plus_explicit_peak_bytes'],'source_conditional_checker_failure_base_plus_explicit_bytes':row['checker_failure']['base_plus_explicit_value_bytes'],'unproven_full_int64_sum_cast_extra_bytes':row['input_B_failure']['backend_sum_int64_full_cast_if_materialized_extra_bytes'],'total_peak_certified':False,'original_gate_unchanged':True})
limits=['Only phase1 queries; no tensor contents read, phase2, numerical smoke or full checker executed.','Historical torch dispatch traces and current queried DSO are separate evidence; source checkout is not asserted equal to SDK binary.','Peak arithmetic is source-conditional explicit tensor payload plus observed workspace, not a total HBM bound. Excludes allocator retention/fragmentation, runtime overhead, index/cast/arithmetic/failure reduction scratch and small failure scalars.','Failure torch.sum(bool) full int64 materialization remains unproven; do not add or remove it as a certified allocation.','Initial NORMAL_ONLY B allocation failed207001 before any phase1 call; preserved. HUGE_FIRST successful allocation does not certify original Torch caching allocator fullcase behavior.','Fullcase gates and root pair ownership unchanged; accepted_added=0.']
result={'status':'phase1_workspace_diagnostics_complete_not_fullcase_success','accepted_added':0,'distinct_IsClose_queries':7,'distinct_All_queries':6,'failed_allocation_attempt_queries':0,'scope':sorted(scope),'checks':checks,'keys':keys,'limits':limits,'blocked_descriptors':[]}
(H/'RESULT.json').write_text(json.dumps(result,indent=2))
delivery={'status':result['status'],'accepted_added':0,'findings':{'B_IsClose_workspace_bytes':16777728,'B_All_workspace_bytes':16796160,'all_queries_rc0':True,'all_successful_guards_rc0':True,'first_B_allocation_failure_preserved':'p00_B/result.json','repair_evidence':'ALLOCATION_CORRECTION.json'},'files':['RESULT.json','PLAN.json','TRACE_EVIDENCE.json','ACTUAL_TRACE_METADATA.json','SOURCE_PINS.json','ALLOCATION_CORRECTION.json','probe.py','probe_normal_only_failed.py','prepare.py','run_probe.py','resource_gate.py','summarize.py','FILEHASHES.json'],'checks':'13 distinct phase1 calls; actual trace metadata, independent device buffers, host/HBM gates, guard identity and executor cleanup verified in summarize.py','remaining_issues':limits,'no_external_writes':True}
(H/'DELIVERY.json').write_text(json.dumps(delivery,indent=2))
(H/'HEARTBEAT.json').write_text(json.dumps({'at':time.time(),'status':'complete','accepted_added':0}))
pins=json.loads((H/'SOURCE_PINS.json').read_text())
for p in [P/'fp16_local_remaining3/environment.sh',Path('/usr/local/bin/npu-guard'),P/'bf16_moe_masked_recovery1/resource_gate.py']:
 pins[str(p.resolve())]=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
(H/'SOURCE_PINS.json').write_text(json.dumps(pins,indent=2))
files={str(p.relative_to(H)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in H.rglob('*') if p.is_file() and p.name!='FILEHASHES.json'}
(H/'FILEHASHES.json').write_text(json.dumps(files,indent=2))
print(json.dumps({'query_workspaces':[(c['id'],[(q['api'],q['workspace_bytes']) for q in c['queries']]) for c in checks],'conditional_success_with_V5':[(k['case_id'],k['source_conditional_success_with_V5_bytes']) for k in keys]},indent=2))
