from pathlib import Path
import json,hashlib,re,datetime
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((R.parent/'attention_generated_recovery5/BUILD_RESUME_EVIDENCE.json').read_text())
art=[]
for d in old['native_artifacts']:
 p=Path(d['path']);h=sha(p);art.append(dict(path=str(p),sha256=h,matches_build_pin=h==d['sha256']))
(R/'ARTIFACT_PINS.json').write_text(json.dumps(art,indent=2))
for row in json.loads((R/'SCOPE.json').read_text()):
 key=row['case_id'];cp=R/(key+'.checker.json')
 if not cp.exists():continue
 c=json.loads(cp.read_text());n=json.loads((R/(key+'.native.json')).read_text());g=json.loads((R/(key+'.guard.json')).read_text())
 assert g['owner']=='baseline700-20261026/attention_direct_api5' and g['phase']=='finished' and g['exit_code']==0
 pid=g['child']['pid'];logpins={};evidence=[];bins=set()
 for p in (R/'runtime_logs'/key).rglob('*'):
  if not p.is_file() or str(pid) not in p.name:continue
  logpins[str(p.relative_to(R))]=sha(p)
  for line in p.read_text(errors='replace').splitlines():
   selected=('OpName:[FlashAttn_' in line and any(x in line for x in ['bin path is','Launch kernel','Launch dynamic kernel','Kernel launch successfully'])) or ('StreamLaunchKernelV2:' in line and 'kernel_name=FlashAttn_' in line) or ('op type [FlashAttnMetadata] run cpu kernel' in line)
   if selected:
    evidence.append(dict(log=str(p.relative_to(R)),line=line))
    if 'bin path is ' in line:bins.add(line.split('bin path is ',1)[1].rstrip('.'))
 n['exact_kernel_binary']={p:sha(Path(p)) for p in bins}
 n['runtime_launch_evidence']=evidence
 (R/(key+'.native.json')).write_text(json.dumps(n,indent=2))
 loaded={p:h for p,h in n['loaded_files_sha256'].items() if 'attention_native_build5_transformer' in p}
 stamps=lambda s:datetime.datetime.fromisoformat(s)
 receipt=dict(case_id=key,status=c['status'],verification=c.get('verification'),inputs_preserved=c.get('inputs_preserved'),input_seed=c.get('input_seed'),checker_receipt=cp.name,guard_receipt=key+'.guard.json',native_receipt=key+'.native.json',maps=key+'.maps',native_launch=evidence,actual_kernel_binaries={p:sha(Path(p)) for p in bins},loaded_vendor_libraries=loaded,all_loaded_vendor_hashes_match=all(sha(Path(p))==h for p,h in loaded.items()),log_sha256=logpins,api_and_adapter_pins=n['frozen_pins'],guard_pure_lock_wait_seconds=(stamps(g['acquired_at'])-stamps(g['requested_at'])).total_seconds(),guard_reserved_seconds=(stamps(g['updated_at'])-stamps(g['acquired_at'])).total_seconds(),checker_wall_seconds=c['wall_seconds'],provider_source_changed=c.get('provider_source_changed'),oracle_source_changed=c.get('oracle_source_changed'))
 (R/(key+'.receipt.json')).write_text(json.dumps(receipt,indent=2))
print('audited',len(art),'artifacts; all match',all(x['matches_build_pin'] for x in art))
