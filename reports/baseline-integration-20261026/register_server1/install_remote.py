import pathlib, subprocess, json
L=pathlib.Path(__file__).parent
ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
source=(L/'verified_where.py').read_text()
script=r'''
import pathlib, json, hashlib, difflib, shutil, ast
W=pathlib.Path('/opt/tileops-all-pairs/workspace')
T=W/'reports/baseline-integration-20261026/register_server1'
P=W/'reports/baseline-integration-20261026/tla_direct'
D=W/'TileOPs/benchmarks/ascend_all_pairs/tla'
T.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
checker=json.loads((P/'runs/256M-float16-pto/checker.json').read_text())
pins=json.loads((P/'DEPLOYMENT_PINS.json').read_text())
pins.update(json.loads((P/'SOURCE_PINS.json').read_text()))
# Restrict source dependencies to the verified Where route.
pins={k:v for k,v in pins.items() if '/examples/normalization/' not in k and '/examples/causal_conv1d/' not in k}
pins.update(checker['source_sha256'])
pins={str(W/k) if not k.startswith('/') else k:v for k,v in pins.items()}
assert all(sha(k)==v for k,v in pins.items())
cfg=dict(key=['tilelang-ascend','WhereFwdOp','256M-float16'],params=checker['case_snapshot']['parameters'],workspace=str(W),vendor=str(W/'ascend_baselines/tilelang-ascend'),compiler=str(W/'.cache/all-pairs/remote_tla_compiler'),runtime=str(W/'.cache/all-pairs/full_rollout_20261024/remote_ready/server1/group1/recovery_resume25/tla_supplemental_1858/runtime_env2/runtime/package'),private=str(P),output_root=str(T/'runtime'),pins=pins)
old=(D/'__init__.py').read_text()
needle='def _make(op,case):\n'
assert old.count(needle)==1
new=old.replace(needle,needle+"    from .verified_where import matches, factory as verified_factory\n    if matches(op, case):\n        return verified_factory(case)\n")
files={'__init__.py':new,'verified_where.py':SOURCE,'verified_where.json':json.dumps(cfg,indent=2)+'\n'}
records=[];patch=[]
for name,data in files.items():
 p=D/name;b=T/'backups'/name;b.parent.mkdir(exist_ok=True)
 before=p.read_text() if p.exists() else ''
 assert not b.exists(), 'Do not overwrite backups'
 if p.exists():shutil.copy2(p,b)
 rec={'path':str(p),'before_sha256':sha(p) if p.exists() else None,'backup':str(b) if p.exists() else None,'after_sha256':hashlib.sha256(data.encode()).hexdigest()}
 records.append(rec)
 patch.extend(difflib.unified_diff(before.splitlines(True),data.splitlines(True),fromfile='a/'+str(p.relative_to(W)) if p.exists() else '/dev/null',tofile='b/'+str(p.relative_to(W))))
 if name.endswith('.py'):compile(data,str(p),'exec')
 p.write_text(data)
(T/'CHANGED_FILES.json').write_text(json.dumps(records,indent=2))
(T/'integration.patch').write_text(''.join(patch))
receipt_names=['checker.json','guard.json','GUARD_RUNNING.json','launch.json','native.json','COMMAND.json']
receipts=[]
for name in receipt_names:
 src=P/'runs/256M-float16-pto'/name
 dst=T/'receipts'/name;dst.parent.mkdir(exist_ok=True);shutil.copy2(src,dst)
 receipts.append({'original':str(src),'copy':str(dst),'sha256':sha(src)})
for name in ['DELIVERY.json','PARENT_REVIEW.json','ACCEPTANCE.json','DEPLOYMENT_PINS.json']:
 src=P/name;dst=T/'receipts'/name
 # PARENT_REVIEW may only be in parent local delivery; copied separately below.
 if src.exists():shutil.copy2(src,dst);receipts.append({'original':str(src),'copy':str(dst),'sha256':sha(src)})
(T/'REGISTRATION.json').write_text(json.dumps(dict(key=cfg['key'],factory='benchmarks.ascend_all_pairs.tla.verified_where.factory',registry='benchmarks.ascend_all_pairs.registry.candidate_factory(case, "tilelang-ascend")()',implementation=str(P/'adapter.py')+':factory',registered_count=1,verified_receipts_reused_count=1,receipts=receipts,remaining_unintegrated_keys=[],new_numerical_runs=0),indent=2))
print(json.dumps({'files':{str(p.relative_to(T)):p.read_text() for p in T.rglob('*') if p.is_file() and p.suffix in ('.json','.patch','.py')},'provider_files':files}))
'''
r=subprocess.run([ssh,'openlibing-DevEnv-661495','python3 -B -'],input='SOURCE='+repr(source)+'\n'+script,text=True,capture_output=True)
if r.returncode:raise RuntimeError(r.stderr+r.stdout)
a=json.loads(r.stdout)
for name,data in a['files'].items():
 p=L/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(data)
for name,data in a['provider_files'].items():
 p=L/'deployed'/name;p.parent.mkdir(exist_ok=True);p.write_text(data)
print('Installed 3 provider files; backups, patch and receipts copied locally')
