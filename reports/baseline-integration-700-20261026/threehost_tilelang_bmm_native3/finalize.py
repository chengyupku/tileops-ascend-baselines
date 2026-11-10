from pathlib import Path
import json,hashlib,time
P=Path(__file__).resolve().parent
scope=json.loads((P/'SCOPE.json').read_text());rows=[]
for item in scope:
 cid=item['case_id'];suffix='_canonical' if cid=='mha-decode-pv-float16' else '';tag='tilelang_BmmFwdOp_'+cid+suffix
 seal=P/(tag+'.SEALED.json');s=json.loads(seal.read_text());c=s['checker'];v=c['verification']
 assert s['key']==[item['baseline'],item['operator'],cid]
 assert c['status']=='verified' and c['inputs_preserved'] and v['checked_outputs']==v['total_outputs']==1
 assert not c['provider_source_changed'] and not c['oracle_source_changed']
 oldtag='tilelang_BmmFwdOp_'+cid+('_v3' if cid=='mha-decode-pv-bfloat16' else '')
 old=json.loads((P.parent/'threehost_tla_other54_r2/server1_evidence/runs'/oldtag/'checker.json').read_text())
 assert c['input_seed']==old['input_seed'] and c['case_snapshot']==old['case_snapshot']
 rows.append(dict(baseline='tilelang',operator='BmmFwdOp',case_id=cid,status='verified',registered=False,acceptance='ready_for_parent_review',host='server1',checker_receipt='server1_evidence/runs/'+tag+'/checker.json',guard_receipt=tag+'.guard.json',native_seal=seal.name,checked_outputs=1,total_outputs=1,inputs_preserved=True,input_seed=c['input_seed'],dispatch_count=s['dispatch_count'],max_abs_err=v['max_abs_err'],checker_wall_seconds=c['wall_seconds'],prior_seed_and_case_snapshot_identical=True,native_artifacts=s['native_checks']))
dep=json.loads((P/'closure/DEPENDENCY_ARTIFACT_HASHES_FINAL.json').read_text())
assert all(r.get('matches_checker',True) for r in dep['files'].values())
old_adapter=P.parent/'threehost_tla_other54_r2/server1_evidence/tilelang_adapter.py'
assert (P/'closure/tilelang_adapter.py').read_bytes()==old_adapter.read_bytes()
(P/'RESULT.json').write_text(json.dumps(rows,indent=2))
delivery=dict(status='complete',scope_count=3,verified_native_complete=3,registered=0,shared_registry_modified=False,canonical_executions=3,guard_wait_timeout_without_execution=2,source_adapter_checker_seed_tolerance_unchanged=True,dependency_files=len(dep['files']),results='RESULT.json',file_hashes='FILEHASHES.json',notes=["Each actual __tvm_ffi_main pointer was captured by RTLD_NOLOAD immediately after TVM load; executable maps inode and saved binary SHA256 were validated.","TVM unlinks temporary exported.so after loading. Every completed dispatch retains the identical pointer-containing mapping/inode with (deleted) suffix; bytes were saved before unlink, never reconstructed from an expected path.","Container stat st_dev=0:61 differs from executable maps fd:03; root overlay mount is recorded in server1_mountinfo.txt. Path, inode, symbol address and mapping ranges are checked directly.","All three physical hosts inspected. Server2 lacked SDK/frozen assets and was occupied; local x86 cannot reuse ARM runtime. Server1 bounded cooperative reservation released before the two remaining canonical executions.","One CPU preflight command found file(1) unavailable; ELF headers were inspected with Python before any device submission.","Parent review/registry integration remains outside worker write scope; baseline stays tilelang."])
(P/'DELIVERY.json').write_text(json.dumps(delivery,indent=2))
files={str(f.relative_to(P)):dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size) for f in sorted(P.rglob('*')) if f.is_file() and f.name!='FILEHASHES.json'}
(P/'FILEHASHES.json').write_text(json.dumps(dict(algorithm='sha256',files=files),indent=2))
print(json.dumps(delivery,indent=2))
