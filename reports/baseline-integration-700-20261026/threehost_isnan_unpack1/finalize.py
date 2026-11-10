from pathlib import Path
import json,hashlib,datetime,ast
P=Path(__file__).resolve().parent;R=P/'runs/tilelang-ascend_IsnanFwdOp_16M-float32_expanded';A=P.parent/'threehost_isnan_adapter1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(P/n).write_text(json.dumps(x,indent=2))
def rel(p):return str(p.relative_to(P.parent))
c=read(R/'checker.json');d=read(R/'1.dispatch.json');b=read(R/'unpack_buffers.json');g=read(P/'guard.json');pre=read(P/'preflight.json');rt=read(R/'actual_runtime_pins.json');launch=read(P/'launcher.json')
assert c['status']=='verified' and c['verification']['checked_outputs']==c['verification']['total_outputs']==1
assert c['inputs_preserved'] and c['verification']['max_abs_err']==0
assert c['input_seed']==1014271450 and not c['oracle_source_changed'] and not c['provider_source_changed']
assert read(R/'final_sync.json')['success'] and d['phase']=='native_sync_completed' and g['returncode']==0
assert all(b[n]['shape']==[2097152,8] and b[n]['stride']==[8,1] and b[n]['contiguous'] for n in ['packed','expanded_shifts'])
assert len({v['ptr'] for v in b.values()})==4
assert sha(P/'original_source.py')==sha(A/'original_source.py')==c['source_sha256']['/opt/tileops-all-pairs/workspace/ascend_baselines/tilelang-ascend/testing/python/language/test_tilelang_ascend_language_elementwise.py']
assert pre['runner_sha256']==sha(P/'run_isnan.py') and pre['adapter_sha256']==sha(P/'element_adapter.py')
assert read(R/'headers_manifest.json')==read(P/'headers_manifest.json')
assert sha(P/'headers_snapshot.tar.gz')==pre['header_snapshot_sha256']
assert sha(R/'native_artifacts'/d['binary_sha256'])==d['binary_sha256']
remote='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_isnan_unpack1'
pins={**pre['pins'],**{n:v['sha256'] for n,v in pre['original_files'].items()},**{n:v['sha256'] for n,v in rt.items() if '/tla_cache/' not in n},**read(P/'headers_manifest.json')}
pins.update({remote+'/element_adapter.py':sha(P/'element_adapter.py'),launch['path']:launch['sha256'],**{n:v['sha256'] for n,v in launch['support'].items()}})
save('provider_pins.json',pins)
contract=dict(exact_key=['tilelang-ascend','IsnanFwdOp','16M-float32'],case_snapshot=c['case_snapshot'],seed=c['input_seed'],reference='Original manifest workloads.elementwise._REFERENCES[IsnanFwdOp], unchanged',verification='Original Exact() and bool dtype tolerance; every element via contiguous 1Mi-element blocks',oracle_source_sha256=c['oracle_source_sha256'],verifier_sha256=c['verifier_sha256'],original_source_sha256=sha(P/'original_source.py'),source_api=c['api'],kernel=dict(api='compare',M=4096,N=4096,block_M=128,block_N=256,mode='NE',dtype='float',out_dtype='uint8',target='pto',platform='A5',pass_configs={'TL_ASCEND_AUTO_CV_SYNC':True,'TL_ASCEND_AUTO_SYNC':True,'TL_ASCEND_MEMORY_PLANNING':True},packed_shape=[4096,512]),binding=dict(input='contiguous float32; same tensor passed twice to original compare',unpack='CPU uint8 [0..7] -> device; both operands expand to [2097152,8] then contiguous; same-shape >> and &1; reshape [4096,4096]; bool conversion',extra_expanded_buffers_bytes=33554432,original_buffers_not_overwritten=True),actual_buffers=b,observed_calls=c['observed_calls'],result=c['verification'],inputs_preserved=True,accepted_added=0)
save('CASE_CONTRACT.json',contract)
checks=dict(cpu_exhaustive=read(P/'CPU_PROOF.json'),lazy_routing=read(P/'STATIC_ROUTING.json'),full_checker_verified=True,checked_outputs=1,total_outputs=1,checked_elements=16777216,input_preservation=True,native_sync=True,final_sync=True,independent_expanded_buffers=True,original_source_unchanged=True,oracle_checker_unchanged=True,header_snapshot_verified_at_compile=True,actual_mapped_runtime_pins=len(rt),guard_exit=0,device_runs=1)
save('CHECKS.json',checks)
result=dict(exact_key=contract['exact_key'],status='verified_unregistered',accepted_added=0,registered=False,host='server2',verification=c['verification'],checked_elements=16777216,inputs_preserved=True,native_phase=d['phase'],final_sync=True,guard_returncode=0,checker_receipt=rel(R/'checker.json'),guard_receipt=rel(P/'guard.json'),case_contract=rel(P/'CASE_CONTRACT.json'),source_api=c['api'],original_source_sha256=sha(P/'original_source.py'),generated_source_sha256=d['generated_sha256'],native_binary_sha256=d['binary_sha256'],wall_seconds=c['wall_seconds'],remaining_issues=['Parent must review and publish; no shared registry changed.','Lazy provider packaging has CPU static routing validation only; full device evidence is the original checker runner using identical adapter.','Deployment is pinned to server2 aarch64/CANN 9.1 runtime plus sdk-920b2 compiler; no cross-host portability claim.'],evidence_paths=[rel(R/n) for n in ['checker.json','1.dispatch.json','1.kernel.txt','compiled.cpp','compiler_command.json','actual_runtime_pins.json','deployment.json','unpack_buffers.json','final_sync.json']])
save('RESULT.json',result)
save('DELIVERY.json',dict(**result,checks=rel(P/'CHECKS.json'),lazy_provider=rel(P/'lazy_provider_proposal.py'),runtime_pins=rel(P/'provider_pins.json'),deployment_recipe=rel(P/'DEPLOYMENT.md'),header_snapshot=rel(P/'headers_snapshot.tar.gz')))
save('HEARTBEAT.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase='complete_verified_unregistered',accepted_added=0,device_runs=1,guard_exit=0))
files={rel(f):sha(f) for f in sorted(P.rglob('*')) if f.is_file() and f.name!='FILEHASHES.json' and '__pycache__' not in f.parts}
save('FILEHASHES.json',files)
print(json.dumps(dict(status=result['status'],checked_elements=16777216,accepted_added=0,files=len(files))))
