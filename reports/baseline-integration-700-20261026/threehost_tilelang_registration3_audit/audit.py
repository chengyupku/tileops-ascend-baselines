"""Read-only historical evidence audit; stdlib only. Writes private reports."""
import json, pathlib, hashlib, collections
O=pathlib.Path(__file__).resolve().parent; B=O.parent; E=B/'threehost_tla_other54_r2/server1_evidence'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,indent=2)+'\n')
p=json.loads((O/'PINS.json').read_text()); inv=json.loads((E/'DEPENDENCY_ARTIFACT_HASHES.json').read_text()); records={};generated=[]
for receipt in p['receipts']:
 r=pathlib.Path(receipt['directory']);load=json.loads((r/'native_load_1.json').read_text());assert sha(r/'native_artifacts'/load['sha256'])==load['sha256']
 dispatches=list(r.glob('*.dispatch.json')); joined=0
 for f in dispatches:
  d=json.loads(f.read_text());assert d.get('completed');assert sha(r/(f.name.split('.')[0]+'.kernel.txt'))==d['kernel_source_sha256']
  lines=[line.split(maxsplit=5) for line in d['maps_after'].splitlines()]; target=[a for a in lines if len(a)==6 and a[5].removesuffix(' (deleted)')==load['path'] and int(a[4])==load['inode']]
  assert target and any(int(a[0].split('-')[0],16)<=load['symbols']['__tvm_ffi_main']<int(a[0].split('-')[1],16) for a in target);joined+=1
  for a in lines:
   if len(a)!=6 or '.so' not in a[5] or not any(x in a[5] for x in ['/torch/','/torch_npu/','/tvm/','/tvm_ffi/','/tilelang/']):continue
   path=a[5];v=records.setdefault(path,{'inodes':set(),'map_devices':set(),'receipts':set()});v['inodes'].add(int(a[4]));v['map_devices'].add(a[3]);v['receipts'].add(str(r.relative_to(B)))
 generated.append({'case':receipt['key'],'binary_sha256':load['sha256'],'load_receipt':str((r/'native_load_1.json').relative_to(B)),'dispatch_joins':joined,'deleted_path_required_to_exist':False,'preload_bytes_and_symbol':True})
for path,v in records.items():
 for k in v:v[k]=sorted(v[k])
 item=inv['files'].get(path);v['inventory']=item;v['historical_hash']=p['runtime'].get(path) if p['runtime_provenance'].get(path)=='successful native.json hash' else None
 v['association']='execution-time hash' if v['historical_hash'] else ('post-run same inode' if item and item.get('inode') in v['inodes'] else 'no supplied content hash')
missing=[k for k,v in records.items() if v['association']=='no supplied content hash']
post={k:v for k,v in p['runtime'].items() if p['runtime_provenance'][k].startswith('post-run')}
compiler='/opt/tileops-ci/runtime/lib/python3.12/site-packages/tilelang/contrib/bisheng.py';local=B.parents[1]/'tilelang/tilelang/contrib/bisheng.py'
preserved=O/'closure'/p['runtime'][compiler]
compiler_match=preserved.exists() and sha(preserved)==p['runtime'][compiler]
if compiler_match:
 dest=O/'closure'/p['runtime'][compiler];assert sha(dest)==p['runtime'][compiler]
report={'scope':'only three successful tilelang GQA2/MoE1 receipts; no whole-machine closure','runtime_library_records':records,'missing_runtime_content_hashes':missing,'post_run_inventory_difference':post,'post_run_count':len(post),'post_run_sdk_count':sum('/sdk-920-release-headers/' in k for k in post),'compiler_source':{'path':compiler,'sha256':p['runtime'][compiler],'remote_read_bytes_match_historical':compiler_match,'invocation_logic':'BISHENG_HOME/bin/bisheng, then PATH; subprocess argv built in compile_ascend','actual_argv_or_header_depfile_captured':False},'sdk_identity':{'receipt':'threehost_tla_other54_r2/SDK_PROVENANCE.txt','layout':'threehost_tla_other54_r2/release_sdk_layout.txt','official_headers_commit':'a63d5261c7c8077a0081e867f43a0e884fec066b','execution_environment':'threehost_tla_other54_r2/server1_evidence/run_case.py:15-16','limitation':'Static environment and SDK identity are supported; compiler executable/header hashes were collected after execution, not by compiler invocation tracing.'},'generated_binaries':generated,'parent_review':'threehost_tla_other54_r2/PARENT_NATIVE3_REVIEW.json','remote_hash_reads':1,'npu_runs':0}
save('EVIDENCE_DIFF.json',report)
gaps=[{'id':'runtime20','count':len(missing),'paths_file':'EVIDENCE_DIFF.json#/missing_runtime_content_hashes','cannot_reconstruct':'Historical bytes/hash of these mapped libraries from maps path/inode alone. Current CPU hash would only establish present inventory.','minimal_resolution':'Locate an existing execution-associated content receipt or immutable package/archive identity for these exact 20 libraries.'},{'id':'compiler_inputs','cannot_reconstruct':'Exact executed compiler argv and actually consumed header set/content from generated binary alone. 679 SDK inventory entries are not 679 proven missing used dependencies.','minimal_resolution':'Locate existing compiler build log/depfile or immutable SDK snapshot with execution association; do not demand unused headers or whole-machine dependency closure.'}]
p['remaining_evidence']=gaps;p['historical_sdk_audit']=report['sdk_identity'];p['publication_ready']=False;p['audit_report']='EVIDENCE_DIFF.json';save('PINS.json',p)
s=O/'registration.py';t=s.read_text();a=t.index('DATA_HASHES =');z=t.index('\n',a);t=t[:a]+'DATA_HASHES = '+repr({n:sha(O/n) for n in ['CONTRACTS.json','PINS.json']})+t[z:];s.write_text(t)
save('PARENT_READY_PROPOSAL.json',{'status':'reviewable_private_package_not_publishable','publication_ready':False,'accepted_added':0,'baseline':'tilelang','scope':[r['key'] for r in p['receipts']],'resolved':['Parent full-checker/guard/296 dispatch audit acknowledged with exact review reference; no pending-parent blanket blocker.','All three preserved exported.so binaries rehashed and joined to completed dispatch maps/inodes/symbols; deleted export paths are not required.','Compiler Python source compared against execution-time hash.','No unused ARM9.1/FP8/BF16 template constants; original math/source bytes preserved.'],'remaining_issues':gaps,'formal_registry_modified':False})
print(json.dumps({'mapped':len(records),'missing':len(missing),'compiler_source_match':compiler_match,'generated_joins':sum(x['dispatch_joins'] for x in generated)}))
