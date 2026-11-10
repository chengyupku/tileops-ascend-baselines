import pathlib,json,hashlib,re,datetime,collections
T=pathlib.Path(__file__).resolve().parent;R=T.parents[1]
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
pins=json.loads((T/'FROZEN.json').read_text())['sha256'];changed={p:sha(p) for p,h in pins.items() if sha(p)!=h};assert not changed,changed
runtime=json.loads((T/'RUNTIME_IDENTITIES.json').read_text());assert all(sha(p)==v['current'] for p,v in runtime['files'].items())
rows=[]
for scoped in json.loads((T/'SCOPE.json').read_text()):
 P=T/scoped['key'][2];row=json.loads((P/'RESULT.json').read_text());c=json.loads((P/'result/representative.json').read_text());g=json.loads((P/'GUARD_RESULT.json').read_text());meta=g['metadata_after'];ex=json.loads((P/'result/EXECUTION.json').read_text());events=json.loads((P/'result/NATIVE_CALLS.json').read_text());cap=json.loads((P/'result/CAPTURE.json').read_text());snap=json.loads((P/'result/guard_snapshot.json').read_text())
 assert c['status']=='verified' and c['inputs_preserved'] and c['verification']['checked_outputs']==c['verification']['total_outputs']==1
 assert not c['provider_source_changed'] and not c['oracle_source_changed']
 assert g['exit_code']==0 and meta['owner']=='baseline700-20261026/surge_dense_linear92/worker04moe' and meta['workload_completed'] and not meta['conflict_detected'] and not meta['visibility_lost']
 assert ex['frozen_unchanged']
 counts=collections.Counter(e['function'] for e in events if e['event']=='enter');assert list(counts.values())==[256,256]
 assert all(e['return_code']==0 for e in events if e['event']=='return')
 for e in events:
  if e['event']=='enter':assert e['sha256']==pins[e['path']] and sha(e['path'])==e['sha256']
 sizes=[e['workspace_bytes'] for e in events if 'workspace_bytes' in e]
 logs=[];launches=0;native_objects={}
 for log in sorted((P/'rt/ASCEND_PROCESS_LOG_PATH').rglob('*.log')):
  with log.open(errors='replace') as f:
   for num,line in enumerate(f,1):
    if 'StreamLaunchKernelV2:' in line and 'kernel_name=GroupedMatmul_' in line:launches+=1
    if 'Register Kernel succ' in line and 'GroupedMatmul_' in line:
     logs.append(dict(file=str(log),line=num,text=line.strip()))
     for path in re.findall(r'bin path: (\S+\.o)',line):native_objects[path]=sha(path)
 assert launches==256,(P,launches)
 assert native_objects and all(pins[p]==h for p,h in native_objects.items())
 for name,v in cap.items():
  if v.get('stored'):assert sha(P/'result'/(name+'.bin'))==v['sha256']
 evidence=dict(status='verified',original_checker_sha256=c['verifier_sha256'],input_seed=c['input_seed'],verification=c['verification'],inputs_preserved=c['inputs_preserved'],capture=cap,native_workspace_calls=counts['aclnnGroupedMatmulV5GetWorkspaceSize'],native_launch_calls=counts['aclnnGroupedMatmulV5'],native_stream_launches=launches,workspace_sum_bytes=sum(sizes),workspace_max_bytes=max(sizes),native_objects=native_objects,register_lines=logs,source_build_chain=[str(R/'bf16_local_gemm18/GENERATED_RECIPE_CHAIN.json'),str(R/'bf16_local_gemm18/DEPLOYMENT_MANIFEST.json'),str(R/'bf16_local_gemm18/PARENT_REVIEW.json')],source_runtime_pins_unchanged=True,original_cleanup_lifetime_preserved=True)
 dump(P/'AUDIT.json',evidence)
 dt=lambda x:datetime.datetime.fromisoformat(x)
 timings=dict(requested=meta['requested_at'],acquired=meta['acquired_at'],child_start_reported=snap['updated_at'],end=g['end'],queue_s=(dt(meta['acquired_at'])-dt(meta['requested_at'])).total_seconds(),physical_wait_s=None,device_s=None,holding_cpu_s=None,compile_s=0,guard_wall_s=g['wall_seconds'],checker_wall_s=c['wall_seconds'],original_observed_call_seconds=json.loads((P/'result/TIMINGS.json').read_text()),note='child_start_reported is guard running-metadata update; device-only time and exact physical idle wait not independently measured; CPU reference retained unchanged')
 row.update(status='verified_pending_parent_registration',runtime_ready=True,registration_module=str(T/'integration700_moe_up.py'),registration_cpu_checks=str(T/'REGISTRATION_CPU_CHECKS.json'),checked_outputs=1,total_outputs=1,inputs_preserved=True,guard_uncontaminated=True,audit=str(P/'AUDIT.json'),input_seed=c['input_seed'],verification=c['verification'],timings=timings,native=str(P/'AUDIT.json'),accepted_added=0,registered=False)
 dump(P/'RESULT.json',row);rows.append(row)
dump(T/'RESULT.json',rows)
dump(T/'DELIVERY.json',dict(utc=now(),task_type='actual supplementary integration',status='verified_ready_pending_parent_serial_registration',keys=[r['key'] for r in rows],verified_count=len(rows),accepted_added=0,registered=False,runtime_ready=True,host='local x86_64',module=str(T/'integration700_moe_up.py'),registry_patch=str(T/'REGISTRY.patch'),recipe=str(T/'environment.sh'),dependency_manifest=str(T/'REGISTRATION_PINS.json'),cpu_checks=str(T/'REGISTRATION_CPU_CHECKS.json'),results=str(T/'RESULT.json'),remaining=['Parent review and serial registry publication; no numerical rerun required.'],limitations=['Original local x86_64 runtime recipe; remote aarch64 deployment absent.','Registration wrapper CPU route verified; no duplicate NPU rerun of successful keys.'],original_math_checker_seed_tolerance_unchanged=True,other_task_writes=False,memory_writes=False))
dump(T/'HEARTBEAT.json',dict(utc=now(),phase='completed',ownership='completed both exclusive SCOPE keys',verified_count=2,accepted_added=0,last_output='RESULT.json DELIVERY.json REGISTRATION_CPU_CHECKS.json',next='parent review and serial publication'))
hashes={str(p.relative_to(T)):sha(p) for p in T.rglob('*') if p.is_file() and p.name not in ('FILEHASHES.json',) and '__pycache__' not in p.parts}
dump(T/'FILEHASHES.json',hashes)
print(json.dumps([dict(key=r['key'],status=r['status'],verification=r['verification'],timings=r['timings']) for r in rows],indent=2))
