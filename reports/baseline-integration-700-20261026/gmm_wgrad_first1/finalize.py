import json,hashlib,re
from pathlib import Path
from datetime import datetime
P=Path(__file__).resolve().parent
load=lambda n:json.loads((P/n).read_text())
dump=lambda n,v:(P/n).write_text(json.dumps(v,indent=2)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
lock=load('guard_snapshot.json');end=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
assert end['owner']==lock['owner'] and end['phase']=='finished' and end['exit_code']==0
dump('GUARD_FINAL.json',end)
r=load('representative.json');e=load('EXECUTION.json');cp=load('REAL_REGISTRY_CPU_PROOF.json');reg=load('REGISTRATION_PROOF.json');caps=load('CAPTURE.json')
assert r['status']=='verified' and e['checker_exit_code']==0 and e['frozen_unchanged']
assert cp['cpu_only'] and not cp['torch_npu_imported'] and len(cp['rows'])==1
preserve={k:caps[k]['sha256']==caps[k.replace('_post_input','_input')]['sha256'] for k in caps if '_post_input' in k}
assert all(preserve.values());dump('INPUT_PRESERVATION.json',preserve)
stages=[json.loads(s) for s in (P/'stages.jsonl').read_text().splitlines()]
secs=lambda a,b:(datetime.fromisoformat(b)-datetime.fromisoformat(a)).total_seconds()
timings={'guard_queue_to_lock_s':secs(lock['requested_at'],lock['acquired_at']),'guard_idle_check_before_child_s':secs(lock['acquired_at'],lock['updated_at']),'guard_child_to_finished_s':secs(lock['updated_at'],end['updated_at']),'guard_total_s':secs(lock['requested_at'],end['updated_at']),'checker_wall_s':r['wall_seconds'],'cpu_reference_compute_s':stages[2]['monotonic']-stages[1]['monotonic'],'candidate_call_s':stages[4]['monotonic']-stages[3]['monotonic'],'note':'Single observed correctness run, candidate time is not a benchmark or isolated kernel timing.'}
dump('TIMINGS.json',timings)
logs=[];native_paths=set()
for f in sorted((P/'ASCEND_PROCESS_LOG_PATH').rglob('*.log')):
 lines=[]
 for line in f.read_text(errors='replace').splitlines():
  if 'private_provider' in line and ('grouped_matmul' in line.lower() or 'GroupedMatmul' in line):
   lines.append(line)
   for s in re.findall(r'/[^\s\[\]"\',;()]+',line):
    if Path(s).is_file():native_paths.add(s)
 if lines:
  rel=f.relative_to(P);out=P/'native_log_extracts'/rel;out.parent.mkdir(parents=True,exist_ok=True);out.write_text('\n'.join(lines)+'\n')
  logs.append({'original_relative_path':str(rel),'original_sha256':sha(f),'extract_relative_path':str(out.relative_to(P)),'matching_lines':len(lines)})
dump('NATIVE_LOAD_EVIDENCE.json',{'logs':logs,'referenced_native_files_sha256':{s:sha(Path(s)) for s in sorted(native_paths)},'host_mappings':load('RUNTIME_MAPS.json')['native_mapped_sha256'],'full_original_logs_retained_remotely':True})
v='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_dense/gemm/private_provider'
result={'key':['ops-transformer','GroupedGemmFwdOp','moe-wgrad-float16'],'status':'verified_registered','accepted_added_by_worker':0,'suggested_acceptance_count':1,'canonical_npu_attempts':1,'guard_exit_code':0,'checker_exit_code':0,'verification':r['verification'],'group_count':16,'output_shape':[16,4096,4096],'checked_output_elements':16*4096*4096,'inputs_preserved':r['inputs_preserved'],'input_seed':r['input_seed'],'registered':True,'factory':reg['factory'],'route_module':reg['module'],'api':r['api'],'original_complete_api':['aclnnGroupedMatmulV5GetWorkspaceSize','aclnnGroupedMatmulV5'],'adaptation':'Canonical metadata slices into 16 independent groups; transpose A/contiguous packing; FP16 inputs losslessly widened to FP32; original full V5 API per group; output cast back to FP16 and stack. No K splitting or kernel changes.','required_runtime_environment':{'ASCEND_CUSTOM_OPP_PATH':v,'ASCEND_CLASSIC_TRANSFORMER_VENDOR':v,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'},'timings':timings,'source_pin':{'before':'FROZEN.json','after':'SOURCE_PIN_AFTER.json','unchanged':True,'checker_provider_unchanged':not r['provider_source_changed'],'checker_oracle_unchanged':not r['oracle_source_changed']},'sync_cleanup':'Original checker finally synchronized and candidate cleanup returned normally; checker and guard exit 0.','native_evidence':'NATIVE_LOAD_EVIDENCE.json','registry_cpu_proof':'REAL_REGISTRY_CPU_PROOF.json','registry_stub_boundary':'Actual on-disk registry and real factory executed; _MathCall constructor stubbed; no numeric run, ACL DSO load or torch_npu import. Numerical proof exclusively original full checker.','changed_shared_files':[reg['module'],str(Path(reg['module']).parent/'__init__.py')],'registration_preserved_prior_routes':reg['previous_routes_preserved'],'remaining_issues':[],'payload_location':str(P),'payload_transfer':'Tensor .bin and full native logs retained remote; metadata, hashes, receipts and relative-path native extracts transferred only.'}
dump('RESULT.json',result)
dump('DELIVERY.json',result|{'evidence':['representative.json','guard.log','guard.exit','GUARD_FINAL.json','guard_snapshot.json','EXECUTION.json','CAPTURE.json','INPUT_PRESERVATION.json','FROZEN.json','SOURCE_PIN_AFTER.json','NATIVE_LOAD_EVIDENCE.json','RUNTIME_MAPS.json','REGISTRATION_PROOF.json','REAL_REGISTRY_CPU_PROOF.json','TIMINGS.json','stages.jsonl'],'remaining':[]})
print(json.dumps({'status':result['status'],'native_logs':logs,'native_files':len(native_paths),'timings':timings}))
