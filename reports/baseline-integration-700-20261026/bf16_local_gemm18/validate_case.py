from pathlib import Path
import json,sys,hashlib
P=Path(sys.argv[1]);T=P.parent
r=json.loads((P/'representative.json').read_text());e=json.loads((P/'EXECUTION.json').read_text());g=json.loads((P/'GUARD_RECEIPT.json').read_text());cap=json.loads((P/'CAPTURE.json').read_text())
assert r['status']=='verified' and r['verification']['checked_outputs']==r['verification']['total_outputs']==1
assert e['frozen_unchanged'] and g['returncode']==0
assert e['capture_counts']=={'reference':1,'candidate':1}
for i in range(2):
 assert len({cap[f'{who}_0_{post}input{i}']['sha256'] for who in ['reference','candidate'] for post in ['', 'post_']})==1
for n,v in cap.items():assert hashlib.sha256((P/(n+'.bin')).read_bytes()).hexdigest()==v['sha256']
lines=[]
for f in (P/'ASCEND_PROCESS_LOG_PATH/debug/plog').glob('*'):
 for line in f.read_text(errors='replace').splitlines():
  if ('GroupedMatmul' in line and any(x in line for x in ['Register Kernel succ','StreamLaunchKernelV2','Leaving function aclnnGroupedMatmulV5','Entering function aclnnGroupedMatmulV5'])):lines.append(line)
(P/'NATIVE_DISPATCH.log').write_text('\n'.join(lines)+'\n')
assert any('Register Kernel succ' in x and str(T/'install/vendors/bf16gmm_transformer') in x and 'e84aaec0' in x for x in lines)
assert any('StreamLaunchKernelV2' in x and 'e84aaec0' in x for x in lines)
assert any('Leaving function aclnnGroupedMatmulV5GetWorkspaceSize' in x for x in lines)
assert any('/install/vendors/bf16gmm_transformer/op_api/lib/libcust_opapi.so' in x for x in e['loaded_files'])
a={'status':'verified_unregistered','checked_outputs':1,'full_inputs_outputs_preserved':True,'input_hashes_equal':True,'actual_native_dispatch':True,'source_frozen':True,'accepted_added':0,'seed':r['input_seed'],'verification':r['verification']}
(P/'VALIDATION.json').write_text(json.dumps(a,indent=2));print(P.name,a['status'])
