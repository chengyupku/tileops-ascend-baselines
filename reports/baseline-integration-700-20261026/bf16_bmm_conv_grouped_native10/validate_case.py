from pathlib import Path
import json,sys,hashlib
P=Path(sys.argv[1]);T=P.parent;D=T.parent/'bf16_local_gemm18'
r=json.loads((P/'representative.json').read_text());e=json.loads((P/'EXECUTION.json').read_text());g=json.loads((P/'GUARD_RECEIPT.json').read_text());cap=json.loads((P/'CAPTURE.json').read_text())
assert r['status']=='verified',r
assert r['verification']['checked_outputs']==r['verification']['total_outputs']==1
assert e['frozen_unchanged'] and g['returncode']==0
assert e['capture_counts']=={'reference':1,'candidate':1},e['capture_counts']
ninputs=len([k for k in cap if k.startswith('candidate_0_input')])
for i in range(ninputs):
 entries=[cap[f'{who}_0_{post}input{i}'] for who in ['reference','candidate'] for post in ['', 'post_']]
 assert len({x.get('sha256',json.dumps(x)) for x in entries})==1,(i,entries)
for n,v in cap.items():
 if v.get('stored'):assert hashlib.sha256((P/(n+'.bin')).read_bytes()).hexdigest()==v['sha256']
lines=[]
for f in (P/'ASCEND_PROCESS_LOG_PATH').rglob('*'):
 if f.is_file():
  for line in f.read_text(errors='replace').splitlines():
   if ('GroupedMatmul' in line and any(x in line for x in ['Register Kernel succ','StreamLaunchKernelV2','Leaving function aclnnGroupedMatmulV5','Entering function aclnnGroupedMatmulV5'])):lines.append(line)
(P/'NATIVE_DISPATCH.log').write_text('\n'.join(lines)+'\n')
assert any('Register Kernel succ' in x and str(D/'install/vendors/bf16gmm_transformer') in x and 'e84aaec0' in x for x in lines),'device registration missing'
assert any('StreamLaunchKernelV2' in x and 'e84aaec0' in x for x in lines),'launch missing'
assert any('Leaving function aclnnGroupedMatmulV5GetWorkspaceSize' in x for x in lines),'workspace API missing'
assert any(x.rstrip().endswith('Leaving function aclnnGroupedMatmulV5') for x in lines),'native launch API missing'
release=json.loads((T/'FROZEN.json').read_text())['release'];host=release['native_host'];assert e['mapped_host_hashes'][host['path']]==host['sha256']
a={'status':'verified_unregistered','checked_outputs':1,'all_outputs_checked':True,'input_hashes_equal':True,'input_count':ninputs,'actual_native_dispatch':True,'source_frozen':True,'accepted_added':0,'seed':r['input_seed'],'verification':r['verification'],'timings':e['timings'],'native_dispatch_lines':len(lines),'mapped_host':host,'device_object':release['device_object']}
(P/'VALIDATION.json').write_text(json.dumps(a,indent=2));print(P.name,a['status'])
