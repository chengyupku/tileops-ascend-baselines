from pathlib import Path
import json,hashlib,datetime
W=Path('/root/dyq/workspace'); P=W/'reports/baseline-integration-700-20261026/nn_binding_recovery2'; R=W/'reports/baseline-integration-700-20261026/nn_remote_remaining6'
def save(n,d): (P/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def evidence(path,ranges):
 p=W/path if not Path(path).is_absolute() else Path(path)
 lines=p.read_text().splitlines()
 return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),excerpts=[dict(start=a,end=b,text='\n'.join(lines[a-1:b])) for a,b in ranges])
records=[]
for f,spans in [
 ('TileOPs/workloads/numerics.py',[(277,302),(330,387),(431,440),(543,565),(575,580)]),
 ('TileOPs/scripts/ascend_pair_check.py',[(376,384)]),
 ('TileOPs/workloads/norm.py',[(259,276),(421,426)]),
 ('TileOPs/benchmarks/ascend_all_pairs/ops_math/graph.py',[(76,78),(138,145),(154,171)]),
 ('TileOPs/workloads/quantization/int8_dequant.py',[(19,23)]),
 (str(R/'case2/adapter.py'),[(67,84)]),
 (str(R/'case5/adapter.py'),[(91,99)]),
 (str(R/'case2/ASCEND_PROCESS_LOG_PATH/debug/plog/plog-2893521_20261026160437204.log'),[(80972,80979),(89731,89731),(91154,91157)])]:
 records.append(evidence(f,spans))
receipts={}
for i in [2,4,5]:
 path=R/f'case{i}/checker.json';d=json.loads(path.read_text())
 receipts[f'case{i}']={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),**{k:d.get(k) for k in ['runtime_host','case_id','input_seed','case_snapshot','verification','status','api','observed_calls','reference_device']}}
 if i==2:
  (P/'DEQUANT_ORIGINAL_TRACEBACK.txt').write_text(d['traceback'])
  receipts['case2']['traceback_copy']='DEQUANT_ORIGINAL_TRACEBACK.txt'
  receipts['case2']['source_hash_match']={f:hashlib.sha256((W/f).read_bytes()).hexdigest()==d['source_sha256'][f] for f in ['TileOPs/workloads/numerics.py','TileOPs/benchmarks/ascend_all_pairs/ops_math/graph.py']}
save('EVIDENCE.json',{'source_evidence':records,'selected_receipts':receipts,'known':[
 'case2 checker actually ran locally on x86_64 hostname aa; SCOPE inherits server2 metadata from an older attempt. Do not apply old host/timing metadata to this receipt.',
 'case2 first traceback is numerics.py:382 diagnostic finite.any; log names IsFinite fault at task 467. Later _restore copy_ raises aclnnInplaceCopy and obscures initial exception.',
 'case2 candidate returned expected BF16 shape; graph retains input/output tensors, descriptors and argument arrays until synchronized cleanup. No premature-free evidence in this path.',
 'case5 candidate and reference both return (BF16 output,None,None); structural None validation succeeds, but checked count ignores empty tensor-pair lists. Parent gate rejects 1 != 3.',
 'case4 has real running-stat tensor inputs/outputs and verified 3/3. It establishes a different output contract, not resnet-stem success.'
 ],'inference_and_limits':[
 'dequant failure is attributable to a verifier diagnostic device task, not evidence of Mul numerical mismatch; preceding async corruption cannot be conclusively excluded from logs.',
 'Reaching diagnostics suggests tolerance checks completed at Python control-flow level, but async execution and incomplete receipt prevent acceptance.',
 'BatchNorm has no binding-only fix: dropping None outputs changes tuple width, replacing them by tensors violates None contract, and changing input/seed/oracle/checker is forbidden.',
 'No adapter mutation is justified for dequant without new evidence; a synchronous diagnostic run could localize an earlier async fault, but is not an acceptance run or authorized worker action.'
 ]})
scope=json.loads((P/'SCOPE.json').read_text())
rows=[]
for s in scope:
 bn=s['operator']=='BatchNormFwdOp'
 rows.append({'baseline':s['baseline'],'operator':s['operator'],'case_id':s['case_id'],'accepted_added':0,'status':'blocked_outside_binding_scope','binding_patch':None,
 'root_cause': 'Coverage counting rejects two structurally validated None outputs (1/3), not missing adapter outputs or demonstrated numerical mismatch.' if bn else 'Verifier diagnostic IsFinite device fault, followed by cleanup input-copy error; original Mul numerical failure is not established.',
 'firstcase_admission':False,
 'admission_requirement': 'Requires separately authorized resolution of None coverage accounting in original validation infrastructure while preserving the complete three-slot contract; forbidden in this task. Do not retry identical adapter.' if bn else 'Require a changed, justified environment/deployment or separately authorized diagnostic/infrastructure resolution with exact original seed/input/oracle preserved. Current binding is unchanged; no identical acceptance rerun.',
 'next_cpu_command':f'python3 -B {P}/cpu_contract_check.py' if bn else f"rg -n 'fault kernel_name=|aclnnInplaceCopy_715' {R}/case2/ASCEND_PROCESS_LOG_PATH/debug/plog/plog-2893521_20261026160437204.log",
 'numerical_success_claimed':False})
save('RESULT.json',rows)
save('DELIVERY.json',{'worker_id':'079c27e3fe294fec','finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accepted_added':0,'scope_keys':len(rows),'outcome':'Two evidence-backed blockers; no legal top-level repair established','changed_files':[str(P/n) for n in ['cpu_contract_check.py','CPU_CHECKS.json','write_delivery.py','DELIVERY.json','RESULT.json','EVIDENCE.json','DEQUANT_ORIGINAL_TRACEBACK.txt']], 'checks':{'cpu_stub_passed':True,'torch_imports':0,'tensor_operations':0,'npu_runs':0,'native_compiles':0,'remote_mutations':0,'shared_writes':0,'agents_spawned':0},'timing':{'elapsed_minutes_estimate':9,'precision':'approximate; start timestamp was not instrumented','scope':'investigation plus artifact preparation'},'next_steps':[r['admission_requirement'] for r in rows],'binding_patch_prepared':False,'reason_no_patch':'Neither issue has a supported binding-only fix. Avoid speculative adapters that change output contracts or conceal verifier failures.'})
