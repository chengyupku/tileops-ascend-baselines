import pathlib,json,hashlib,datetime
P=pathlib.Path(__file__).resolve().parent;E=P/'server1_evidence'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(P/n).write_text(json.dumps(v,indent=2)+'\n')
proof=read(P/'PROOF.json');freeze=read(E/'FREEZE.json');pins=read(P/'PINS.json')
gaps=[dict(id='host_compiler_pre_freeze',step='CPU preparation before first guarded checker',path='/usr/bin/g++',detail='g++ was not included in initial CPU frozen set. All actual g++ subprocesses have original argv/env/cwd and immediate pre-exec content SHA; TOOL_POSTFREEZE confirms matching inode/hash and preserved bytes afterwards. This establishes execution-associated content identity, but does not satisfy the requested outside-guard pre-freeze procedure.',resolution='Parent can review the explicit preparation deviation. No new checker run performed or proposed here; all three authorized runs already succeeded once.')]
pins['previous_receipts']=pins['receipts'];pins['previous_remaining_evidence']=pins['remaining_evidence']
pins['receipts']=[dict(key=r['key'],directory=str((E/'runs'/'_'.join(r['key'])).resolve()),checker_sha256=sha(P/r['checker']),guard_sha256=sha(P/r['guard']),native_sha256=sha(E/'runs'/'_'.join(r['key'])/'native_load_1.json')) for r in proof['rows']]
# Retain prior lazy full-contract implementation and immutable original source closure.
for p,r in freeze['files'].items():
 if '/threehost_tilelang3_complete_capture/frozen/' in p:continue
 pins['runtime'][p]=r['sha256'];pins['runtime_provenance'][p]='fresh execution-associated frozen snapshot; pre/post content match; PROOF.json'
for r in read(E/'TOOL_POSTFREEZE.json'):
 if pathlib.Path(r['path']).name=='g++':pins['runtime'][r['path']]=r['sha256'];pins['runtime_provenance'][r['path']]='immediate pre-exec SHA + matching post-freeze bytes/inode; NOT outside-guard pre-frozen'
pins['publication_ready']=False;pins['remaining_evidence']=gaps;pins['audit_report']='PROOF.json';pins['fresh_proof_sha256']=sha(P/'PROOF.json');pins['accepted_added']=0
save('PINS.json',pins)
f=P/'registration.py';s=f.read_text();a=s.index('DATA_HASHES =');z=s.index('\n',a);s=s[:a]+'DATA_HASHES = '+repr({n:sha(P/n) for n in ['CONTRACTS.json','PINS.json']})+s[z:];s=s.replace('Historical dependency closure audit pending; execution disabled','Host compiler pre-freeze procedure deviation remains for publication review; execution disabled');f.write_text(s)
f=P/'prepare_run.py';f.write_text(f.read_text().replace('baseline700/tilelang-registration3','baseline700-20261026/threehost_tilelang3_complete_capture'))
summary=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='three_fresh_full_checkers_verified_publication_held_for_preparation_deviation',accepted_added=0,publication_ready=False,scope=3,full_checker_passes=3,completed_dispatches=296,native_binaries_preserved=3,mapped_runtime_paths=29,previous_runtime20_gaps_resolved=20,frozen_files=2871,pre_post_hash_mismatches=0,original_math_bytes_unchanged=True,original_checker_seed_reference_tolerance_unchanged=True,npu_runs=3,reruns=0,shared_registry_writes=0,memory_writes=0,remaining_issues=gaps,proof='PROOF.json',cpu_checks='CPU_CHECK.json',module='registration.py',pins='PINS.json',contracts='CONTRACTS.json',remote_frozen_content='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_tilelang3_complete_capture/frozen_content',old_success_preserved='../threehost_tla_other54_r2/PARENT_NATIVE3_REVIEW.json',scope_note='Only GQA2/MoE1 ran. A local SCOPE.json containing six rows was observed after execution and restored from the immutable three CONTRACTS; observed bytes retained in SCOPE_observed_before_restore.json. Remote scope and three guard receipts were always the authorized three.')
save('DELIVERY.json',summary);save('RESULT.json',summary)
save('HEARTBEAT.json',dict(utc=summary['utc'],phase='completed',accepted_added=0,full_checker_passes=3,publication_ready=False))
(P/'RECIPE.md').write_text('''# Private TileLang GQA2/MoE1 complete capture

Original mathematical implementation and five source closure bytes are unchanged. Three original full checkers ran once under foreground npu-guard on server1, with separate fresh caches. No accepted cases were repeated and accepted_added=0. PROOF.json rederives 296 native joins, all 29 mapped library identities, original complete verification equality, and the 2,871-file pre/post snapshot checks.

The original lazy full-contract registration.py, CONTRACTS.json, closure/, prepare_run.py and CPU checks are retained. The private module is deployment-layout-ready but execution remains deliberately gated by PINS.publication_ready=false. The only remaining procedural deviation is /usr/bin/g++ missing from the initial CPU snapshot: actual subprocess interception hashed it before execution, and matching inode/hash/bytes were preserved after execution. Bisheng, ld.lld and original official SDK header directories were pre-frozen. No compiler flags or SDK paths were replaced. Do not claim the g++ post-freeze happened before execution.

CPU checks: PYTHONDONTWRITEBYTECODE=1 python3 cpu_checks.py
Proof audit: python3 verify_proof.py
Do not rerun launch.py: the authorized single capture is complete.

server1_evidence/ contains full checker/native/dispatch/compiler/mapping/pre-post receipts. Content-addressed frozen files remain in the same private directory on server1; FREEZE.json records paths and hashes. Remote and local clocks differ by about 94 seconds; use same-host timestamps for ordering. Shared registry and parent memory are untouched.
''')
print(json.dumps({k:summary[k] for k in ['full_checker_passes','completed_dispatches','previous_runtime20_gaps_resolved','publication_ready','accepted_added']}))
