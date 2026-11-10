from pathlib import Path
import json,hashlib,subprocess,gzip,re,datetime
W=Path('/root/dyq/workspace');D=W/'reports/baseline-integration-700-20261026';P=D/'gmm_fp16_remaining11';RW='/opt/tileops-all-pairs/workspace';RP=RW+'/reports/baseline-integration-700-20261026/gmm_fp16_remaining11'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
remote={};local_count=0;native_count=0;qualified=[];remote_native=[]
for k,h in read(P/'FINAL_MANIFEST.json')['sha256'].items():assert sha(P/k)==h,k
rows=[x for x in read(P/'RESULT.json')['results'] if x.get('recommended_acceptance')];assert len(rows)==7
registry=read(P/'REAL_REGISTRY_CPU_PROOF.json');assert len(registry['rows'])==7 and not registry['torch_npu_imported']
for x in rows:
 q=(P/x['checker_receipt']).parent;c=read(q/'representative.json');key=x['key'];assert [c['library'],c['op'],c['case_id']]==key
 assert sha(q/'representative.json')==x['checker_sha256'];assert c['status']=='verified' and c['inputs_preserved'] and not c['provider_source_changed'] and not c['oracle_source_changed']
 v=c['verification'];assert v['checked_outputs']==v['total_outputs']>0 and v['unchecked_reason'] is None
 assert c['input_seed']==int.from_bytes(hashlib.sha256((key[1]+'::'+key[2]).encode()).digest()[:4],'big')
 assert (q/'guard.exit').read_text().strip()=='0'
 e=read(q/'EXECUTION.json');assert e['checker_exit_code']==0 and e['frozen_unchanged'] and e['capture_counts']=={'reference':1,'candidate':1}
 for k,h in c['oracle_source_sha256'].items():assert sha(W/k)==h,k;remote[RW+'/'+k]=h
 remote[RW+'/TileOPs/scripts/ascend_pair_check.py']=c['verifier_sha256']
 for k,h in c['source_sha256'].items():remote[k if k.startswith('/') else RW+'/'+k]=h
 frozen=read(q/'FROZEN.json')['sha256'];art=read(q/'REMOTE_ARTIFACTS.json')
 for k,info in art.items():
  f=q/k
  if f.is_file():assert sha(f)==info['sha256'],str(f);local_count+=1
  if not k.endswith('.bin'):remote[RP+'/'+q.name+'/'+k]=info['sha256']
 cap=read(q/'CAPTURE.json')
 for k,info in cap.items():
  if k.startswith('reference_0_input'):
   ix=k.split('input')[1]
   for label in ['reference_0_post_input','candidate_0_input','candidate_0_post_input']:
    other=cap[label+ix];assert all(info.get(f)==other.get(f) for f in ['shape','dtype','sha256']);assert ('sha256' in info)==('sha256' in other)
 audit=next(t for t in read(P/'CHECK_AUDIT.json') if t['attempt']==q.name)
 raw='\n'.join(gzip.open(z,'rt',errors='replace').read() for z in q.glob('*.log.gz'))
 for z in audit['actual_native_bins']:
  assert frozen[z['path']]==z['sha256'];remote[z['path']]=z['sha256'];native_count+=1
  if z['driver_line'] not in raw:remote_native.append({'root':RP+'/'+q.name+'/ASCEND_PROCESS_LOG_PATH','line':z['driver_line']})
  tm=re.search(r'(2026-10-26-\d\d:\d\d:\d\d)\.(\d+)\.(\d+)',z['driver_line']);assert tm
  ts=tm[1].replace('2026-10-26-','2026-10-26T')+'.'+tm[2]+tm[3]+'+00:00'
  assert datetime.datetime.fromisoformat(c['started_at'])<=datetime.datetime.fromisoformat(ts)<=datetime.datetime.fromisoformat(c['finished_at'])
 assert any('/private_provider/op_api/lib/libcust_opapi.so' in f for f in e['loaded_files'])
 proof=read(P/'REGISTRATION_PROOF.json' if q.name=='conv02_fp32' else q/'REGISTRATION_PROOF.json')
 assert proof['key']==key and proof['checker_sha256']==x['checker_sha256'] and proof['prior_registry_prefix_preserved']
 live=next(t for t in registry['rows'] if t['key']==key);assert proof['tested_binding_sha256'] in live['actual_source_sha256'].values()
 remote[proof['tested_binding']]=proof['tested_binding_sha256']
 qualified.append({'key':key,'checker':str(q/'representative.json'),'guard':str(q/'guard.exit'),'verification':v,'guard_returncode':0,'host':'server1','original_checker_status':'verified'})
original=read(D/'audit_dense33/ORIGINAL_SOURCE_CHECK.json')[0]['files']
for z in original:assert z['matches'] and sha(z['path'])==z['sha256'];remote[z['installed']]=z['sha256']
module=RW+'/TileOPs/benchmarks/ascend_all_pairs/ops_transformer/integration700_gmm_fp16_remaining11.py';remote[module]=sha(P/'final_provider_module.py');remote[registry['registry_path']]=registry['registry_sha256']
initial=read(P/'REMOTE_STATIC.json')[registry['registry_path']]['text'];assert (P/'final_registry.py').read_text().startswith(initial)
remote[RP+'/registry_cpu_check.py']=sha(P/'registry_cpu_check.py')
code="""import pathlib,hashlib,json,sys,fcntl
sys.dont_write_bytecode=True
lock=open('/tmp/baseline700-provider-ops_transformer.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
hashes=HASHES
for p,h in hashes.items():
 assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h,p
native_matches=[]
for event in NATIVE:
 matched=[]
 for f in pathlib.Path(event['root']).rglob('*.log'):
  if event['line'] in f.read_text(errors='replace'):matched.append({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
 assert matched,event
 native_matches.append(matched)
print(json.dumps({'remote_hashes_verified':len(hashes),'remote_original_native_log_matches':native_matches}))
""".replace('HASHES',repr(remote)).replace('NATIVE',repr(remote_native))
# Same reviewed CPU-only registry/factory proof, emit to stdout instead of changing worker evidence.
s=(P/'registry_cpu_check.py').read_text();s=s[:s.index("(P/'REAL_REGISTRY_CPU_PROOF.json').write_text")]+"print(json.dumps(out))\n"
code+=s
ssh=str(W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh')
r=subprocess.run([ssh,'openlibing-DevEnv-661495','TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1 /opt/tileops-ci/runtime/bin/python -S -'],input=code,text=True,capture_output=True,timeout=55)
(P/'PARENT_LIVE_CHECK.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr[-3000:]
lines=[json.loads(z) for z in r.stdout.splitlines() if z.startswith('{')];live=lines[-1];assert len(live['rows'])==7 and not live['torch_npu_imported'];assert {tuple(z['key']) for z in live['rows']}=={tuple(z['key']) for z in rows}
(P/'PARENT_LIVE_REGISTRY.json').write_text(json.dumps(live,indent=2)+'\n')
review={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'qualified_rows':qualified,'local_artifact_hashes_verified':local_count,'remote_source_binary_receipt_hashes_verified':len(remote),'original_source_files_matched':len(original),'native_binary_load_events_matched':native_count,'fresh_actual_factory_routes':7,'original_boundary':'Original full GroupedMatmulV5 FP32 native API; lossless FP16 input widening/output cast, convolution packing and native bias, metadata-defined independent MoE groups, existing fixed Hadamard matrix coefficient convention. No kernel/compiler/target/oracle/tolerance edits.','runtime_environment':read(P/'DELIVERY.json')['required_runtime_environment'],'qualification':'MoE raw tensor payloads retained remote; parent verified frozen capture metadata/input hash equality and source/native receipts, without downloading duplicate large payloads. Constructor stubs check routing only; numerical evidence comes from original full canonical checker.','new_npu_runs':0,'remaining':read(P/'DELIVERY.json')['remaining'],'accepted_added':0,'status':'eligible_pending_ledger_commit'}
(P/'PARENT_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n');print({k:v for k,v in review.items() if k not in ['qualified_rows','remaining']})
