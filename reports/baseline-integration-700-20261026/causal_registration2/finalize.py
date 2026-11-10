import json,hashlib,runpy
from pathlib import Path
D=Path(__file__).resolve().parent; B=D.parent/'causal_native_build2'
def write(n,x):(D/n).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=runpy.run_path(str(D/'integration700_causal_registration2.py'))
result=json.loads((B/'RESULT.json').read_text())
source=json.loads((B/'SOURCE_PINS.json').read_text())
pins=m['PINS']; conflicts=[]
for row in source:
 p=str(B/'source'/row['path'])
 if pins.get(p)!=row['copy_sha256']:conflicts.append(p)
assert not conflicts
closure=json.loads((B/'DEPLOYMENT_CLOSURE.json').read_text())
# Original adapter pins are the dependency authority, not locally substituted libraries.
local_checked=[]; remote_required=[]
for p,h in pins.items():
 f=Path(p)
 if f.is_file():
  assert sha(f)==h,p
  local_checked.append(p)
 else:remote_required.append(p)
checks=json.loads((D/'CPU_CHECKS.json').read_text())
checks.update(dependency_count=len(pins),local_files_hash_verified=len(local_checked),server2_only_dependencies=len(remote_required),source_pin_records_crosschecked=len(source))
write('CPU_CHECKS.json',checks)
recipe=json.loads((B/'adapter/RUNTIME_RECIPE.json').read_text())
plan={
 'status':'ready_for_parent_review_and_locked_publication','accepted_added':0,
 'module':'integration700_causal_registration2.py','append_snippet':'registration.append.py',
 'publication_target':'/root/dyq/workspace/TileOPs/benchmarks/ascend_all_pairs/ops_transformer',
 'publication_owner':'parent only; hold ops_transformer publication lock; copy module then append snippet once; fresh CPU interpreter check_published.py',
 'exactkeys':['|'.join(k) for k in m['CASE_PARAMS']],
 'canonical_params':{'|'.join(k):json.loads(v) for k,v in m['CASE_PARAMS'].items()},
 'routing':'Exact baseline/op/case AND all canonical parameters. Known key with altered/missing/extra params raises UnsupportedCase. Unknown case delegates unchanged previous factory; unrelated providers unchanged.',
 'adapter_policy':'Execute original verified binding.py/direct_acl.py bytes without modification; private builtins import routing and frozen Path reads; original __file__; no shared module cache mutation.',
 'state_policy':(B/'STATE_LAYOUT.txt').read_text().strip(),
 'runtime':{
  'host':'server2 only, fresh process; local CPU registry must not load aarch64 DSOs',
  'recipe':recipe,
  'environment_authority':str(B/'check_child.sh'),
  'original_environment_shell':(B/'check_child.sh').read_text(),
  'guard':'Original direct_acl.require_guard unchanged: GDN_CASE_OWNER nonempty equals running metadata.owner; metadata.child.pid == os.getpgrp(). Use original baseline700-20261026/causal_native_build2 owner and npu-guard foreground reservation; no broadened checks.',
  'launcher':'Prepare original check_child.sh environment and recipe python_paths before runtime imports. Use registry on target checkout instead of adapter entry.py monkeypatch. Never rerun successful numerical cases merely for registration.',
  'evidence':'Set GDN_RUN_DIR to a new validation root outside causal_native_build2. Each invocation creates unique child directory and original binding writes NATIVE_LAUNCH.json/loaded.maps there through private os.environ; shared environment unchanged.',
  'lifecycle':'Candidate creation CPU-only. First run guard, all dependency hashes, mapped-library check, then lazy original binding.build. Every invocation guard/hash/pre-map/post-map; cleanup guard and original cleanup. host_roundtrip retained.',
  'loaded_identity':'All 8 previously observed vendor/CANN libraries pinned by realpath and sha256; /proc/self/maps dev+inode must match current stat; unexpected same-basename paths/deleted/missing post-run mappings reject. Pre-first-call allows not-yet-loaded DSOs, rejects any conflicting mappings. Subsequent calls require all observed DSOs.',
 },
 'pins':{'count':len(pins),'local_verified':local_checked,'server2_required':remote_required,'source_pins_crosschecked':len(source)},
 'validation':checks,
 'prior_numeric_evidence':{'result':str(B/'RESULT.json'),'cases':[{k:c[k] for k in ['exactkey','checked_outputs','total_outputs','max_abs_err','inputs_preserved','full_api_state_shape','checker','launch_evidence']} for c in result['cases']], 'review_owner':'parent independently verifies fullchecker/native/source evidence; worker did not rerun'},
 'remaining_issues':['Parent server2 dependency hash recheck and original numerical/source audit remain external to this CPU-only worker.','Shared publication and fresh published-registry check remain parent-owned; no shared files changed.']}
write('REGISTRATION_PLAN.json',plan)
write('RESULT.json',{'status':'cpu_verified_registration_ready_pending_parent_publication','accepted_added':0,'registered':False,'exactkeys':plan['exactkeys'],'cpu_checks_passed':True,'npu_runs':0,'real_torch_acl_imports':0,'native_builds':0,'numerical_reruns':0,'original_adapter_bytes_modified':False,'prior_numeric_cases':2,'prior_numeric_outputs':4,'remaining_issues':plan['remaining_issues']})
write('DELIVERY.json',{'status':'ready_for_parent_review','accepted_added':0,'module':str(D/'integration700_causal_registration2.py'),'append_snippet':str(D/'registration.append.py'),'plan':str(D/'REGISTRATION_PLAN.json'),'cpu_checks':str(D/'CPU_CHECKS.json'),'published_check':str(D/'check_published.py'),'pins':str(D/'PINS.json'),'result':str(D/'RESULT.json'),'shared_writes':0})
write('FILEHASHES.json',{p.name:sha(p) for p in sorted(D.iterdir()) if p.is_file() and p.name not in ['FILEHASHES.json','SPAWN.json','TASK.txt','SCOPE.json']})
print(json.dumps({'status':'ready','dependencies':len(pins),'local_hash_verified':len(local_checked),'server2_required':len(remote_required),'source_pins_crosschecked':len(source)}))
