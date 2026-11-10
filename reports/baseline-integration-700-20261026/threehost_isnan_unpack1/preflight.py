from pathlib import Path
import hashlib,json,shutil,tarfile,platform,subprocess,os
P=Path(__file__).resolve().parent;A=P.parent/'threehost_isnan_adapter1';W=P.parents[2];V=W/'ascend_baselines/tilelang-ascend';C=W/'.cache/all-pairs/remote_tla_compiler';S=W/'.cache/all-pairs/remote_native_deepgemm_deepep/sdk-920b2'
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
for n in ['runtime','frozen','private_tla']:
 if not (P/n).exists():shutil.copytree(A/n,P/n,ignore=shutil.ignore_patterns('__pycache__'))
prior=json.loads((A/'runs/tilelang-ascend_IsnanFwdOp_16M-float32_template_fixed/deployment.json').read_text())
for name,e in prior['files'].items():assert sha(Path(name))==e['sha256'],name
pins={str(f):sha(f) for root in [P/'runtime',P/'frozen',P/'private_tla'] for f in root.rglob('*') if f.is_file()}
assert sha(V/'testing/python/language/test_tilelang_ascend_language_elementwise.py')==prior['source_sha256']
headers={}
roots=[V/'src',V/'3rdparty/pto-isa/include',S/'include',S/'pkg_inc',Path('/usr/local/Ascend/driver/kernel/inc')]
with tarfile.open(P/'headers_snapshot.tar.gz','w:gz') as t:
 for root in roots:
  assert root.is_dir(),root
  for f in sorted(root.rglob('*')):
   if f.is_file():headers[str(f)]=sha(f);t.add(f,arcname=str(f).lstrip('/'),recursive=False)
(P/'headers_manifest.json').write_text(json.dumps(headers,indent=2))
required=[V/'3rdparty/tvm/python/tvm/__init__.py',C/'python',W/'.cache/all-pairs/remote_native_tla/python-deps',S/'tools/bisheng_compiler/bin/bisheng']
assert all(f.exists() for f in required)
for n in ['frozen/scripts/ascend_pair_check.py','frozen/workloads/numerics.py','frozen/workloads/elementwise.py','runtime/tilelang/jit/kernel.py']:assert sha(P/n)==sha(A/n)
result=dict(arch=platform.machine(),previous_deployment_hashes_verified=True,pins=pins,original_files=prior['files'],source_sha256=prior['source_sha256'],required_paths=[str(f) for f in required],header_files=len(headers),header_snapshot_sha256=sha(P/'headers_snapshot.tar.gz'),adapter_sha256=sha(P/'element_adapter.py'),runner_sha256=sha(P/'run_isnan.py'))
(P/'preflight.json').write_text(json.dumps(result,indent=2))
r=subprocess.run(['npu-guard','status','--json'],capture_output=True,text=True);(P/'health_initial.json').write_text(r.stdout);print(json.dumps(dict(preflight='passed',headers=len(headers),runtime_files=len(pins),status_returncode=r.returncode,status=json.loads(r.stdout))))
