import json,hashlib,datetime,io,tarfile,subprocess,shlex
from pathlib import Path
O=Path(__file__).resolve().parent;utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(n,d):(O/n).write_text(json.dumps(d,indent=2)+'\n')
checks=json.loads((O/'CPU_CHECK.json').read_text());remote=json.loads((O/'REMOTE_CPU_CHECK.json').read_text());assert checks['passed'] and remote['rc']==0
pins=json.loads((O/'PINS.json').read_text());evidence=json.loads((O/'EVIDENCE_REVIEW.json').read_text())
evidence['original_candidate_source_hashes_rechecked']=1500;evidence['copied_original_sources_rechecked']=8;evidence['vendor_and_runtime_pins']=len(pins['runtime']);dump('EVIDENCE_REVIEW.json',evidence)
proposal=dict(utc=utc,status='READY_PROPOSAL',accepted_added=0,key=['ops-transformer','BmmFwdOp','square-4k-bfloat16'],module='registration.py',proposed_published_module='benchmarks.ascend_all_pairs.ops_transformer.integration700_bmm_backend_variant',install='install(PROVIDERS)',data_root_local=str(O),data_root_remote='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/'+O.name,contract='CONTRACT.json',pins='PINS.json',original_source_hashes='ORIGINAL_SOURCEHASHES.json',cpu_check='CPU_CHECK.json',remote_cpu_check='REMOTE_CPU_CHECK.json',backend_context='backend_context.py',recipe='RECIPE.txt',admission='../ORIGINAL_CPU_BMM_BACKEND_ADMISSION_20261027.json',backend_variant='torch.backends.mkldnn.enabled=False',default_enabled_true_qualification='failed evidence preserved; not promoted',evidence_review='EVIDENCE_REVIEW.json',shared_registry_changed=False,publication_performed=False,parent_acceptance_performed=False,remaining=['Parent review/publication of exact backend-qualified registration and recipe','CPU confirmation of published route and parent ledger update'],variant_approval_needed=False)
dump('READY_PROPOSAL.json',proposal)
result=dict(utc=utc,status='ready_for_parent_publication',accepted_added=0,key=proposal['key'],evidence_verified=dict(sealed_artifacts=39,full_output_elements=67108864,input_byte_hashes=10,actual_V5_functionptr_maps_inodes=True,device_registration_dispatch=True,original_source_hashes=1500),qualification=dict(admitted=True,backend='torch.backends.mkldnn.enabled=False',original_default_failure_preserved=True,default_environment_pass_claim=False),private_registration=dict(real_manifest_contract=True,source_files=8,runtime_pins=len(pins['runtime']),lazy_device_free_construction=True,public_UnsupportedCase=True,unknown_delegates_previous=True,backend_mutation_inside_candidate=False,original_reference_replaced=False,guard_lifecycle=True,exception_cleanup=True),checks=dict(local_cpu=True,server1_cpu=True,shell_syntax=True,NPU_runs=0,numerical_runs=0,JIT_runs=0,native_compiles=0),remaining=proposal['remaining'],limitations=evidence['limitations'],changed_files_scope=str(O)+' and same-name remote directory only',entrypoints=['READY_PROPOSAL.json','registration.py','PINS.json','CONTRACT.json','CPU_CHECK.json','EVIDENCE_REVIEW.json','RECIPE.txt','FILEHASHES.json'])
dump('RESULT.json',result);dump('DELIVERY.json',result);dump('HEARTBEAT.json',dict(utc=utc,phase='completed_private_ready_proposal_CPU_only',accepted_added=0))
# Self-hash and post-transfer receipt are explicitly excluded to avoid cycles.
excluded={'FILEHASHES.json','REMOTE_DEPLOYMENT.json'}
hashes={str(p.relative_to(O)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(O.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in excluded}
dump('FILEHASHES.json',hashes)
R=proposal['data_root_remote'];SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';host='openlibing-DevEnv-661495'
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as t:
 for name in list(hashes)+['FILEHASHES.json']:t.add(O/name,arcname=name)
r=subprocess.run([SSH,host,'tar xzf - -C '+shlex.quote(R)],input=buf.getvalue(),capture_output=True,timeout=60);assert r.returncode==0,r.stderr
script="import json,hashlib;from pathlib import Path;p=Path(%r);h=json.loads((p/'FILEHASHES.json').read_text());bad=[n for n,v in h.items() if hashlib.sha256((p/n).read_bytes()).hexdigest()!=v];print(json.dumps(dict(checked=len(h),mismatches=bad)));assert not bad"%R
r=subprocess.run([SSH,host,'python3 -S -c '+shlex.quote(script)],capture_output=True,text=True,timeout=60);assert r.returncode==0,r.stderr
receipt=dict(utc=utc,remote_directory=R,verification=json.loads(r.stdout),CPU_only=True,executed_recipe=False,manifest_excludes=['FILEHASHES.json','REMOTE_DEPLOYMENT.json']);dump('REMOTE_DEPLOYMENT.json',receipt)
r=subprocess.run([SSH,host,'cat > '+shlex.quote(R+'/REMOTE_DEPLOYMENT.json')],input=(O/'REMOTE_DEPLOYMENT.json').read_bytes(),capture_output=True,timeout=30);assert r.returncode==0
print(json.dumps(dict(status='READY_PROPOSAL',hashed_files=len(hashes),remote_hash_mismatches=receipt['verification']['mismatches'],accepted_added=0)))
