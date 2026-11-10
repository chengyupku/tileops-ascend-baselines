import json,pathlib,hashlib,datetime,ast
P=pathlib.Path(__file__).resolve().parent;S=P.parent/'threehost_pto_original23';now=datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def save(n,d):(P/n).write_text(json.dumps(d,indent=2)+'\n')
src=read(S/'RESULT.json');ev={tuple(r['key']):r for r in read(P/'EVIDENCE.json')};contracts={(r['op'],r['case_id']):r for r in read(P/'CONTRACTS.json')}
links={tuple(r['key']):r for r in read(S/'PRESTART_LINK.json')}
rows=[];manifest=[]
for r in src['rows']:
 k=tuple(r['key']);c=read(pathlib.Path(r['checker']));g=read(pathlib.Path(r['guard']))
 assert g['returncode']==0 and c['status']=='verified' and c['inputs_preserved']
 assert c['input_seed']==r['input_seed']
 assert c['verification']['checked_outputs']==c['verification']['total_outputs']
 link=links[k]
 assert link['generated_cpp_sha256']==link['kernel_source_sha256']
 for a in r['artifacts']:
  assert hashlib.sha256(pathlib.Path(a['elf']).read_bytes()).hexdigest()==a['elf_sha256']
  for d in a['device_sections']:
   assert hashlib.sha256(pathlib.Path(d['path']).read_bytes()).hexdigest()==d['sha256']
 factory=ev[k]['existing_factory'];relative=factory.split(':')[0].replace('.','/')+'.py'
 present=(S/'server1_evidence/frozen'/relative).is_file()
 manifest.append(dict(key=list(k),contract=contracts[k[1:]],input_seed=r['input_seed'],original_api=next(x['original_api'] for x in read(S/'REGISTRATION_PROPOSAL.json')['rows'] if tuple(x['key'])==k),evidence=ev[k],prior_factory_in_frozen_package=present,prior_factory_in_current_local_public_tree=(P.parents[2]/'TileOPs'/relative).is_file(),route='registration.install(tla.PROVIDERS); exact key -> lazy make -> unchanged closure/adapter.py:factory -> compile_pto(target=pto, original pass_configs)'))
 rows.append(dict(key=list(k),status='cpu_registration_proposal_complete',prior_verification='full_checker_native_verified',checker=r['checker'],guard=r['guard'],host=r['host'],reused=True,new_device_attempts=0,accepted_added=0,registration_performed=False,remaining='parent final evidence review and serialized publish/ledger'))
save('MANIFEST.json',dict(task_type='actual_integration_top_level_CPU_only',baseline='tilelang-ascend',owner='baseline700-20261026/threehost_pto_registration23',allowed_writes=[str(P),'/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_pto_registration23'],numerical_remaining=[],registration_remaining=[r['key'] for r in rows],rows=manifest))
save('RESULT.json',dict(utc=now,total=23,cpu_registration_proposals=23,reused_verified=23,new_device_attempts=0,accepted_added=0,rows=rows))
for p in P.glob('*.py'):ast.parse(p.read_text())
save('CHECKS.json',dict(utc=now,original_archive_hashes=read(P/'SOURCE_HASH_CHECK.json'),local_cpu=read(P/'CPU_CHECK.json'),host_preflight={h:read(P/('PREFLIGHT_'+h+'.json'))['passed'] for h in ['local','server1','server2']},remote_cpu_exit={h:read(P/('CPU_ROUTE_'+h+'.json'))['exit'] for h in ['server1','server2']},reused_fullchecker_guard_inputs_seed_and_native_artifact_checks=23,syntax='all private Python AST parsed; no compilation',NPU_runs=0,shared_provider_writes=0,ledger_writes=0,memory_writes=0))
save('DELIVERY.json',dict(utc=now,status='CPU_REGISTRATION_PROPOSAL_COMPLETE_PENDING_PARENT_PUBLICATION',accepted_added=0,success_count=23,new_device_attempts=0,entry='registration.py:install',manifest='MANIFEST.json',recipe='RECIPE.md',result='RESULT.json',checks='CHECKS.json',runtime_ready='CPU deployment and route checked on three hosts; new wrapper device execution intentionally not run',existing_factory='Both named factories exist in original frozen benchmark package, absent from current local TileOPs tree; historical factories bind obsolete runtime and are not silently reused as entrypoints. Original byte-identical adapter is reused.',remaining=['Parent final review of original evidence and this CPU-only overlay','Parent serialized publication/shared registry and unique acceptance accounting'],timing=dict(start_utc='2026-10-27T03:30:00+00:00',end_utc=now,start_precision='approximate from assigned task metadata',cpu_preparation_seconds=None,remote_transport_seconds=None,requested=None,acquired=None,child_start=None,device_end=None,guard_exit=None,queue_seconds=None,device_seconds=0),commands=['python3 -B prepare.py','python3 -B build_contracts.py','python3 -B cpu_checks.py','python3 -I -S -B preflight.py','python3 -B deploy.py','python3 -B finalize.py']))
save('HEARTBEAT.json',dict(utc=now,phase='completed_cpu_delivery',current_exact_keys=[r['key'] for r in rows],last_output='23 complete contracts, unchanged original adapter; threehost CPU preflight and actual provider route checks pass',blocker='Parent publication/review only; no device tests authorized',next='Parent final batch review and serialized registry/ledger publication',accepted_added=0))
save('FILEHASHES.json',dict(algorithm='sha256',files={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in P.rglob('*') if p.is_file() and p.name not in ['FILEHASHES.json','HEARTBEAT.json'] and '__pycache__' not in p.parts}))
print(json.dumps(dict(proposals=23,accepted_added=0,checks='CHECKS.json',delivery='DELIVERY.json')))
