from pathlib import Path
import json,hashlib,subprocess,re,datetime
W=Path('/root/dyq/workspace');D=W/'reports/baseline-integration-700-20261026';P=D/'gmm_wgrad_first1';RW='/opt/tileops-all-pairs/workspace';RP=RW+'/reports/baseline-integration-700-20261026/gmm_wgrad_first1'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=read(P/'ARTIFACT_SHA256.json');assert all(sha(P/k)==v for k,v in m.items())
c=read(P/'representative.json');d=read(P/'DELIVERY.json');key=d['key'];assert [c['library'],c['op'],c['case_id']]==key
assert c['status']=='verified' and c['inputs_preserved'] and not c['oracle_source_changed'] and not c['provider_source_changed'];assert c['verification']['checked_outputs']==c['verification']['total_outputs']==1 and c['verification']['unchecked_reason'] is None
assert c['input_seed']==int.from_bytes(hashlib.sha256((key[1]+'::'+key[2]).encode()).digest()[:4],'big')
g=read(P/'GUARD_FINAL.json');assert g['exit_code']==0 and g['phase']=='finished' and g['owner']=='baseline700-20261026/gmm_wgrad_first1';assert (P/'guard.exit').read_text().strip()=='0'
e=read(P/'EXECUTION.json');assert e['checker_exit_code']==0 and e['frozen_unchanged']
f=read(P/'FROZEN.json')['sha256'];a=read(P/'SOURCE_PIN_AFTER.json');assert a['all_frozen_unchanged'] and f==a['sha256']
cap=read(P/'CAPTURE.json')
for n,v in cap.items():
 if n.startswith('reference_0_input'):
  for prefix in ['reference_0_post_input','candidate_0_input','candidate_0_post_input']:
   t=cap[prefix+n.split('input')[1]];assert all(v.get(k)==t.get(k) for k in ['shape','dtype','sha256'])
assert cap['candidate_0_output']['shape']==[16,4096,4096] and cap['candidate_0_output']['dtype']=='torch.float16'
assert sha(P/'remaining_binding.py')==sha(D/'gmm_fp16_remaining11/remaining_binding.py');assert sha(P/'v5_fp32_bias.py')==sha(D/'gmm_fp16_remaining11/v5_fp32_bias.py')
remote={RP+'/'+k:v for k,v in m.items() if k not in ['TASK.txt','SPAWN.json','ARTIFACT_SHA256.json','HEARTBEAT.json']};remote={}
for k,h in c['source_sha256'].items():remote[k if k.startswith('/') else RW+'/'+k]=h
for k,h in c['oracle_source_sha256'].items():assert sha(W/k)==h;remote[RW+'/'+k]=h
remote[RW+'/TileOPs/scripts/ascend_pair_check.py']=c['verifier_sha256']
for name in ['representative.json','GUARD_FINAL.json','guard.exit','CAPTURE.json','EXECUTION.json','remaining_binding.py','v5_fp32_bias.py']:remote[RP+'/'+name]=sha(P/name)
native=read(P/'NATIVE_LOAD_EVIDENCE.json');remote.update(native['referenced_native_files_sha256']);remote.update(native['host_mappings']);lines=[];loads=0
for item in native['logs']:
 remote[RP+'/'+item['original_relative_path']]=item['original_sha256'];text=(P/item['extract_relative_path']).read_text();lines.append({'path':RP+'/'+item['original_relative_path'],'lines':text.splitlines()})
 for line in text.splitlines():
  if 'Register Kernel succ' not in line:continue
  tm=re.search(r'(2026-10-26-\d\d:\d\d:\d\d)\.(\d+)\.(\d+)',line);assert tm
  ts=tm[1].replace('2026-10-26-','2026-10-26T')+'.'+tm[2]+tm[3]+'+00:00'
  assert datetime.datetime.fromisoformat(c['started_at'])<=datetime.datetime.fromisoformat(ts)<=datetime.datetime.fromisoformat(c['finished_at']);loads+=1
  for b in re.findall(r'(/\S+\.o)',line):assert b in f and f[b]==native['referenced_native_files_sha256'][b]
assert loads>0
original=read(D/'audit_dense33/ORIGINAL_SOURCE_CHECK.json')[0]['files']
for item in original:assert sha(item['path'])==item['sha256'];remote[item['installed']]=item['sha256']
reg=read(P/'REGISTRATION_PROOF.json');proof=read(P/'REAL_REGISTRY_CPU_PROOF.json');assert reg['previous_routes_preserved'] and reg['registry_before_sha256']==sha(P/'registry_before.py');assert reg['tested_binding_sha256']==sha(P/'remaining_binding.py')
remote[reg['module']]=reg['module_sha256'];remote[proof['registry_path']]=reg['registry_after_sha256'];assert sha(P/'integration700_gmm_wgrad_first1.py')==reg['module_sha256']
code="""import pathlib,json,hashlib,fcntl,sys
sys.dont_write_bytecode=True
lock=open('/tmp/baseline700-provider-ops_transformer.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
hashes=HASHES
for p,h in hashes.items():assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h,p
for e in LINES:
 raw=pathlib.Path(e['path']).read_text(errors='replace')
 for line in e['lines']:assert line.rstrip() in raw,line
assert pathlib.Path(REGPATH).read_text().startswith(pathlib.Path(BEFORE).read_text())
""".replace('HASHES',repr(remote)).replace('LINES',repr(lines)).replace('REGPATH',repr(proof['registry_path'])).replace('BEFORE',repr(RP+'/registry_before.py'))
s=(P/'registry_cpu_check.py').read_text();s=s[:s.index("(P/'REAL_REGISTRY_CPU_PROOF.json').write_text")]+"print(json.dumps(out))\n";code+=s
ssh=W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
r=subprocess.run([str(ssh),'openlibing-DevEnv-661495','TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1 /opt/tileops-ci/runtime/bin/python -S -'],input=code,text=True,capture_output=True,timeout=55)
(P/'PARENT_LIVE_CHECK.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr[-3000:];live=json.loads(r.stdout);assert len(live['rows'])==1 and live['rows'][0]['key']==key and not live['torch_npu_imported'];(P/'PARENT_LIVE_REGISTRY.json').write_text(json.dumps(live,indent=2)+'\n')
review={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'key':key,'status':'eligible_pending_ledger_commit','local_artifact_hashes':len(m),'remote_hashes':len(remote),'original_source_files_matched':len(original),'native_load_events_in_checker_interval':loads,'fresh_registry_factories':1,'guard_exit_code':0,'checked_groups':16,'checked_elements':268435456,'original_boundary':'Same complete original GroupedMatmulV5 API and previously audited binding; 16 metadata-defined groups, transpose/packing, FP16-to-FP32 input widening and FP16 output. No kernel/oracle/tolerance change.','timings':d['timings'],'qualification':'Fullrawtensorpayload and fullnative logs retainedremote; verified receipt, capture hashes, native original log membership/hash and fresh exact factory. No rerun of numericalcase.','runtime_environment':d['required_runtime_environment'],'accepted_added':0,'new_npu_runs':0}
(P/'PARENT_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n');print(review)
