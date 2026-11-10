from pathlib import Path
import json,hashlib,time
P=Path(__file__).resolve().parent
run='runs/tilelang-ascend_IsnanFwdOp_16M-float32_template_fixed'
c=json.loads((P/run/'checker.json').read_text());n=json.loads((P/run/'1.dispatch.json').read_text());old=json.loads((P/'prior_receipt_selected.json').read_text())
assert c['input_seed']==old['input_seed']==1014271450
assert c['verifier_sha256']==old['verifier_sha256']
assert {k.split('/',1)[1]:v for k,v in c['oracle_source_sha256'].items()}=={k.split('/',1)[1]:v for k,v in old['oracle_source_sha256'].items()}
assert c['observed_calls']['reference'][0]['outputs']['elements']==16777216
assert n['phase']=='native_sync_completed'
assert json.loads((P/run/'final_sync.json').read_text())['success']
checks=dict(cpu_all_2048_bits=True,source_bytes_unchanged=True,checker_hash_unchanged=True,oracle_hashes_unchanged=True,seed_unchanged=True,full_reference_elements=16777216,native_input_elements=[x['numel'] for x in n['inputs']],native_inputs_same_pointer=n['inputs'][0]['ptr']==n['inputs'][1]['ptr'],public_target=n['target'],actual_arch='dav-c310',actual_memory='REGISTER_BASE',native_sync_completed=True,final_sync_completed=True,checker_success=False,checked_outputs=0,post_candidate_input_preservation_check_completed=False)
(P/'CHECKS.json').write_text(json.dumps(checks,indent=2))
result=dict(exact_key=['tilelang-ascend','IsnanFwdOp','16M-float32'],status='blocked',reason='Original full reference and PTO/A5 native call completed and synchronized; unpack aclnnRightShift rejects broadcast output shape [2097152,1], expected [2097152,8].',accepted_added=0,registered=False,checked_outputs=0,total_outputs=16777216,host='server2',npu_used=True,guard_receipt='guard_template_fixed.json',checker_receipt=run+'/checker.json',native_receipt=run+'/1.dispatch.json',prior_507015='Not reproduced in this PTO/A5 execution; old failure remains a separate historical observation. No general backend repair or numerical success claimed.',attempts=[dict(guard='guard.json',phase='build',native_called=False,reason='Private runtime migration lacked TL_TEMPLATE_PATH; deployment environment fixed with original vendor src path.'),dict(guard='guard_template_fixed.json',phase='verify/unpack',native_called=True,reason=c['error'])],changes=['Private element_adapter.py: original uint8 CPU static constants moved to original device; public target pto as requested.','Private run_isnan.py: A5 deployment, private caches, original frozen checker/provider binding, native dispatch and sync evidence.','Copied original runtime/checker/reference/private loader bytes into task-private directory; no shared changes.'],remaining_issues=['Full checker has no successful candidate output or equality checks.','Right-shift broadcast binding remains unsupported in this torch_npu deployment; no additional device rerun performed.','Post-candidate input-preservation checks did not complete because unpack failed.'],evidence=['CPU_PROOF.json','adapter.diff','server1_inventory.txt','server2_inventory.txt','server2_preflight.json','health_initial.json','postflight.json',run+'/deployment.json',run+'/compiler_command.json',run+'/1.kernel.txt','compiled.cpp',run+'/native_artifacts/'+n['binary_sha256']],checks=checks)
(P/'RESULT.json').write_text(json.dumps(result,indent=2))
(P/'HEARTBEAT.json').write_text(json.dumps(dict(phase='complete_blocked',lastprogress=time.time(),accepted_added=0,reason=result['reason']),indent=2))
delivery=dict(status='blocked',accepted_added=0,registered=False,result='RESULT.json',checks='CHECKS.json',hashes='FILEHASHES.json',heartbeat='HEARTBEAT.json',summary=result['reason'],changed_files=result['changes'],remaining_issues=result['remaining_issues'])
(P/'DELIVERY.json').write_text(json.dumps(delivery,indent=2))
(P/'DELIVERY.md').write_text("""# IsnanFwdOp / 16M-float32
accepted_added=0; registered=false; blocked.

CPU proof covers all 256 bytes × 8 bits and extracts constants from the actual adapter AST. Original compare source SHA256 remains 303b90f72b7b73b800f252e46cebf2dca682bea6b9362dd051f53d80044afb84. Tiles 128×256, pass configs, float32 input, uint8 packed output, bool result, seed 1014271450, checker and reference bytes are unchanged. Only private binding/deployment files were modified.

Three hosts were inspected. Local and server1 were occupied; compatible aarch64 compiler/SDK on idle server2 was used. The original Python deployment was migrated into the task-private remote directory. First build stopped before reference/native execution because template discovery did not follow the relocated package. Setting public TL_TEMPLATE_PATH to existing original vendor src resolved this deployment issue.

The subsequent guarded full-size checker reached the original PTO kernel, using actual bisheng dav-c310 / REGISTER_BASE (A5). Native call pointer, maps, inode, binary, generated source, compile command and SDK/compiler hashes are captured. Full original reference returned 16,777,216 boolean elements. Native call and final sync succeeded; 507015 was not reproduced in this particular run.

Candidate unpack then failed: aclnnRightShift error 161002, output shape [2097152,1] versus required [2097152,8]. Thus zero output comparisons completed; input-preservation checks after candidate return were not completed. This is not a successful integration or general backend repair. No further device retry, registry edit, memory edit, source/kernel modification, or delegation occurred.

See RESULT.json, CHECKS.json, CPU_PROOF.json, guard_template_fixed.json and runs/tilelang-ascend_IsnanFwdOp_16M-float32_template_fixed/. Remaining work is a separately reviewed broadcast-binding repair and successful full checker.
""")
files={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in P.rglob('*') if f.is_file() and f.name!='FILEHASHES.json'}
(P/'FILEHASHES.json').write_text(json.dumps(files,indent=2))
print(json.dumps(dict(files=len(files),status=result['status'],accepted_added=0,checks=checks),indent=2))
