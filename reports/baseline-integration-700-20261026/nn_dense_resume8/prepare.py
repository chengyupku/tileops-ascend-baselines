from pathlib import Path
import shutil,json,hashlib
P=Path(__file__).resolve().parent;W=Path('/root/dyq/workspace');T=W/'TileOPs';S=P/'snapshot'/'TileOPs'
for name in ['benchmarks','workloads','src','scripts']:
 shutil.copytree(T/name,S/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
for name,src in [('adapter.py',W/'reports/baseline-integration-700-20261026/nn_recovery02/adapter.py'),('native_log_capture.py',W/'reports/baseline-integration-20261026/nn_misc_math/recovery01/native_log_capture.py')]:shutil.copy2(src,P/name)
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert h(P/'adapter.py')=='037f902f6779aa1d84dad1d5d04777e6754cec5e9b3d653b7339ac5f0dd200d7'
manifest={str(p.relative_to(S)):h(p) for p in S.rglob('*') if p.is_file()}
(P/'SNAPSHOT.json').write_text(json.dumps(manifest,indent=2))
old=json.loads((W/'reports/baseline-integration-700-20261026/nn_recovery02/fp32_m1_01/checker.json').read_text())
checks={k:h(W/k)==v for k,v in old['oracle_source_sha256'].items()};assert all(checks.values()),checks
native={k:h(W/k)==v for k,v in old['source_sha256'].items() if k.startswith('.cache/ascend-classic/') or k.startswith('ascend_baselines/ops-nn/')}
assert all(native.values()),[k for k,v in native.items() if not v]
(P/'PRECHECK.json').write_text(json.dumps({'original_oracle_matches':checks,'native_and_source_matches':native,'adapter_sha256':h(P/'adapter.py'),'snapshot_files':len(manifest)},indent=2))
print('snapshot files',len(manifest),'native/source checked',len(native))
