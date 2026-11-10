from pathlib import Path
import json,hashlib,datetime
O=Path(__file__).resolve().parent;S=O.parent/'threehost_bmm_backend_full1_r2'
def read(n):return json.loads((S/n).read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
seal=read('server1/SEALED_HASHES.json');bad=[]
for p,h in seal.items():
 if sha(S/'server1'/p)!=h:bad.append(p)
assert len(seal)==39 and not bad
rep=read('server1/square-4k-bfloat16/representative.json');exe=read('server1/square-4k-bfloat16/EXECUTION.json');ih=read('INPUTHASH.json');native=read('server1/square-4k-bfloat16/NATIVE_CALLS.json');proof=read('device_proof/PROOF.json');frozen=read('server1/FROZEN.json')['sha256']
assert rep['status']=='verified' and rep['inputs_preserved'] and not rep['provider_source_changed'] and not rep['oracle_source_changed']
assert exe['checker_exit_code']==0 and exe['frozen_unchanged']
assert rep['observed_calls']['candidate'][0]['outputs']['elements']==67108864
assert rep['verification']['checked_outputs']==rep['verification']['total_outputs']==1
assert all(len(set(g.values()))==1 for g in ih['new_run_input_groups'].values())
for g in ih['new_run_input_groups'].values():
 for name,h in g.items():assert sha(S/'server1/square-4k-bfloat16'/(name+'.bin'))==h
for ev in native:
 if ev['event']=='return':assert ev['return_code']==0
 else:
  fields=ev['map_line'].split();lo,hi=[int(x,16) for x in fields[0].split('-')]
  assert lo<=ev['address']<hi and ev['map_inode']==ev['stat_inode']==int(fields[4])
  assert ev['sha256']==frozen[ev['path']]==exe['mapped_host_hashes'][ev['path']]
assert {x['function'] for x in native}=={'aclnnGroupedMatmulV5','aclnnGroupedMatmulV5GetWorkspaceSize'}
for f in proof['files']:
 assert sha(S/'device_proof'/Path(f['path']).name)==f['sha256']==frozen[f['path']]
assert any('Register Kernel succ' in x for x in proof['launch_log_lines'])
obj=next(f for f in proof['files'] if f['path'].endswith('.o'));stem=Path(obj['path']).stem
assert all(stem in x for x in proof['launch_log_lines']) and any('StreamLaunchKernelV2' in x for x in proof['launch_log_lines'])
backend=read('server1/square-4k-bfloat16/BACKEND_VARIANT.json');events=backend['events'];assert events[-1]['mkldnn_enabled']==backend['original']==True
assert all(x['bmm_unchanged'] for x in events)
assert all(x['mkldnn_enabled'] is False for x in events if x['phase'] in ['reference_before','reference_after','configured'])
assert all(x['reference_source'].strip().endswith('return torch.bmm(a, b)') for x in events if 'reference_source' in x)
old=O.parent/'threehost_bmm_input_backend1/DELIVERY.json';olddata=json.loads(old.read_text())
review=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='evidence_verified_cpu_only',sealed_artifacts=39,seal_mismatches=bad,checked_output_elements=67108864,seed=rep['input_seed'],input_hashes_verified=10,old_input_byte_equality_claim=False,original_source_frozen_count=len(frozen),workspace_and_launch_real_functionptr_maps_inodes_hashes=True,device_registration_to_dispatch=True,device_binary_sha256=obj['sha256'],backend_variant='torch.backends.mkldnn.enabled=False',backend_restored=True,original_BF16_torch_bmm_reference_unchanged=True,atol=.016,rtol=.016,default_enabled_true_failure=dict(path=str(old),sha256=sha(old),status=olddata.get('status'),preserved=True,automatically_promoted=False),accepted_added=0,remaining=['Parent review and publication of private registration and backend evaluation recipe; published-route CPU confirmation.'],limitations=['Historical evidence review only; no new NPU/numeric/JIT/native compilation executed.','Registration proposal CPU route is tested separately; existing fullchecker ran original adapter, not this new wrapper.'])
(O/'EVIDENCE_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n');print(json.dumps({k:review[k] for k in ['status','sealed_artifacts','checked_output_elements','input_hashes_verified','accepted_added']}))
