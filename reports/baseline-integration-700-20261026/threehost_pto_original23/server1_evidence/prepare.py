from pathlib import Path
import sys,shutil,json,hashlib,platform,subprocess,importlib.util
P=Path(__file__).resolve().parent;W=P.parents[2];V=W/'ascend_baselines/tilelang-ascend';C=W/'.cache/all-pairs/remote_tla_compiler';R=W/'.cache/all-pairs/full_rollout_20261024/remote_ready/server1/group1/recovery_resume25/tla_supplemental_1858/runtime_env2/runtime/package'
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
records=[]
def cp(a,b):
 b.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a,b);records.append(dict(original=str(a),private=str(b),sha256=sha(b)))
for root,dest in [(W/'TileOPs/benchmarks',P/'frozen/benchmarks'),(W/'TileOPs/workloads',P/'frozen/workloads'),(W/'TileOPs/src/tileops/manifest',P/'frozen/src/tileops/manifest'),(R/'tilelang',P/'runtime/tilelang'),(W/'reports/baseline-integration-700-20261026/tla_simple_tilelang/frozen_gemm_rest7/private_tla',P/'private_tla')]:
 assert root.exists(),str(root)
 for f in root.rglob('*'):
  if f.is_file() and '__pycache__' not in f.parts and '.cycache' not in f.parts:cp(f,dest/f.relative_to(root))
cp(W/'TileOPs/scripts/ascend_pair_check.py',P/'frozen/scripts/ascend_pair_check.py')
cp(W/'reports/baseline-integration-700-20261026/tla_simple_tilelang/frozen_gemm_rest7/adapter.py',P/'adapter.py')
for rel in ['examples/batch_gemm/batch_gemm.py','examples/gemm/example_gemm.py']:cp(V/rel,P/'original'/rel)
original=json.loads((P/'original_hashes.json').read_text())
checks=[]
for rel,h in original.items():
 f=P/'runtime'/rel if rel.startswith('tilelang/') else P/'original'/rel
 if f.exists():checks.append(dict(path=str(f),sha256=sha(f),original_sha256=h,matches=sha(f)==h))
assert all(x['matches'] for x in checks),[x for x in checks if not x['matches']]
for f in (C/'handoff/lib').glob('*.so'):
 st=f.stat();records.append(dict(original=str(f),sha256=sha(f),inode=st.st_ino,elf_machine=int.from_bytes(f.read_bytes()[18:20],'little')))
sdk=W/'.cache/all-pairs/remote_native_deepgemm_deepep/sdk-920b2';assert sdk.exists()
for f in [sdk/'tools/bisheng_compiler/bin/bisheng',V/'tilelang/jit/adapter/libgen.py']:
 assert f.exists(),str(f);records.append(dict(original=str(f),sha256=sha(f)))
for root in [sdk/'include',sdk/'pkg_inc',V/'3rdparty/pto-isa/include',V/'src']:
 for f in root.rglob('*'):
  if f.is_file():records.append(dict(original=str(f),sha256=sha(f)))
report=dict(host=platform.node(),machine=platform.machine(),python=sys.version,stdlib=sys.path,freeze=records,original_checks=checks,sdk=str(sdk),npu_smi=subprocess.run(['npu-smi','info'],capture_output=True,text=True).stdout)
(P/'PRESTART_FREEZE.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(host=report['host'],files=len(records),original_checks=len(checks),python=sys.version)))
