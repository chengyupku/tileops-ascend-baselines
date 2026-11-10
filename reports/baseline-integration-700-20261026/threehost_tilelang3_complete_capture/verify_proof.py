import pathlib,json,hashlib,datetime
P=pathlib.Path(__file__).resolve().parent;E=P/'server1_evidence'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
freeze=read(E/'FREEZE.json');paths=read(P/'RUNTIME_PATHS.json');rows=[]
for scope in read(P/'SCOPE.json'):
 key=[scope['baseline'],scope['operator'],scope['case_id']];tag='_'.join(key);run=E/'runs'/tag
 c=read(run/'checker.json');assert c['status']=='verified' and c['inputs_preserved']
 guard=read(P/(tag+'.guard.json'));assert guard['returncode']==0
 pre=read(E/('pre_'+tag+'.json'));post=read(E/('post_'+tag+'.json'));assert not pre['changed'] and not post['changed']
 start=read(run/'identity_start.json');assert start['freeze_sha256']==sha(E/'FREEZE.json');assert (run/'identity_end.json').exists()
 runtime=read(run/'runtime_mappings.json')['records'];assert len(runtime)==len(paths)==29
 bypath={r['path']:r for r in runtime};assert set(bypath)==set(paths)
 for p,r in bypath.items():assert r['sha256']==freeze['files'][p]['sha256']==pre['files'][p]['sha256']==post['files'][p]['sha256']
 load=read(run/'native_load_1.json');assert sha(run/'native_artifacts'/load['sha256'])==load['sha256']
 ds=list(run.glob('*.dispatch.json'));joins=0
 for f in ds:
  d=read(f);assert d.get('completed');assert sha(run/(f.name.split('.')[0]+'.kernel.txt'))==d['kernel_source_sha256']
  maps=[s.split(maxsplit=5) for s in d['maps_after'].splitlines()];native=[a for a in maps if len(a)==6 and a[5].removesuffix(' (deleted)')==load['path'] and int(a[4])==load['inode']]
  assert native and any(int(a[0].split('-')[0],16)<=load['symbols']['__tvm_ffi_main']<int(a[0].split('-')[1],16) for a in native)
  for p,r in bypath.items():
   expected=r['maps'][0].split(maxsplit=5);assert any(len(a)==6 and a[5]==r['stat']['realpath'] and a[4]==expected[4] and a[3]==expected[3] for a in maps),p
  joins+=1
 subs=[json.loads(s) for s in (run/'compiler_subprocess.jsonl').read_text().splitlines()];comp=[r for r in subs if r['executable'] and pathlib.Path(r['executable']).name in ['bisheng','ld.lld','g++']]
 assert any(pathlib.Path(r['executable']).name=='bisheng' for r in comp)
 for r in comp:
  assert r['environment']['BISHENG_HOME']==str(pathlib.Path(freeze['header_roots'][0]).parents[1]/'tools/bisheng_compiler')
  assert r['cwd'].endswith(tag)
  assert '-g' not in r['argv'] and '-O0' not in r['argv'] and '-MMD' not in r['argv']
  assert r.get('frozen_executable') or r.get('observed_executable')
 old=next(r for r in read(P.parent/'threehost_tla_other54_r2/PARENT_NATIVE3_REVIEW.json')['rows'] if r['key']==key)
 previous=read(pathlib.Path(old['checker']))
 for name in ['verifier_sha256','input_seed','case_snapshot','oracle_source_sha256','verification']:
  assert c[name]==previous[name],name
 rows.append(dict(key=key,checker=str((run/'checker.json').relative_to(P)),guard=tag+'.guard.json',status=c['status'],dispatch_joins=joins,native_binary_sha256=load['sha256'],mapped_runtime_paths=len(runtime),all_runtime_pre_post_content_match=True,compiler_subprocesses=len(comp),compiler_evidence=str((run/'compiler_subprocess.jsonl').relative_to(P)),original_seed_reference_checker_tolerance_unchanged=True,full_output_verification=c['verification'],wall_seconds=c['wall_seconds']))
proof=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=3,accepted_added=0,rows=rows,full_checker_passes=len(rows),dispatch_joins=sum(r['dispatch_joins'] for r in rows),runtime20_content_gaps_resolved=20,snapshot_files=len(freeze['files']),header_roots=freeze['header_roots'],fresh_caches=True,original_math_bytes_unchanged=True,old_numerical_success_preserved=True,previous_parent_review='../threehost_tla_other54_r2/PARENT_NATIVE3_REVIEW.json',clock_note='Remote UTC is approximately 94 seconds behind local UTC; compare execution ordering within each host only.',preparation_deviation='Bisheng/ld.lld and official SDK headers pre-frozen outside guard; host g++ was discovered by actual subprocess observer, hashed immediately before execution inside guard, then bytes preserved and same inode/hash verified after all runs. It was not pre-frozen outside guard.')
(P/'PROOF.json').write_text(json.dumps(proof,indent=2));print(json.dumps({k:v for k,v in proof.items() if k in ['full_checker_passes','dispatch_joins','runtime20_content_gaps_resolved','snapshot_files']}))
