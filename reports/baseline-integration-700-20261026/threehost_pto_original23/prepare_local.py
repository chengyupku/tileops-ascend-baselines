from pathlib import Path
import shutil,json,hashlib,importlib.machinery,sys
P=Path(__file__).resolve().parent;W=P.parents[2];C=W/'.cache/all-pairs/full_rollout_20261024/repairs/resume_20261025/boundary_1557/tla_pristine_build';V=C/'source';D=P/'local_evidence';D.mkdir(exist_ok=True)
for name in ['frozen','private_tla','original']:
 shutil.copytree(P/'server1_evidence'/name,D/name,dirs_exist_ok=True)
shutil.copytree(V/'tilelang',D/'runtime/tilelang',ignore=shutil.ignore_patterns('__pycache__','.cycache','*.so'),dirs_exist_ok=True)
shutil.copyfile(V/'VERSION',D/'runtime/VERSION')
for name in ['adapter.py','SCOPE.json','original_hashes.json']:shutil.copyfile(P/'server1_evidence'/name,D/name)
shutil.copyfile(P/'observe.py',D/'observe.py')
s=(P/'run_case.py').read_text();s=s.replace("W=Path('/opt/tileops-all-pairs/workspace');V=W/'ascend_baselines/tilelang-ascend'","W=Path('/root/dyq/workspace');V=W/'.cache/all-pairs/full_rollout_20261024/repairs/resume_20261025/boundary_1557/tla_pristine_build/source'")
s=s.replace("C=W/'.cache/all-pairs/remote_tla_compiler'","C=V.parent")
s=s.replace("TILELANG_LIBRARY_PATH=str(C/'handoff/lib'),TVM_LIBRARY_PATH=str(C/'handoff/lib')","TILELANG_LIBRARY_PATH=str(V/'build'),TVM_LIBRARY_PATH=str(V/'build/tvm')")
s=s.replace("str(C/'handoff/bin')","str(W/'.venv/bin')")
s=s.replace("str(C/'python'),str(W/'.cache/all-pairs/remote_native_tla/python-deps')","str(C/'venv/lib/python3.12/site-packages'),str(W/'.venv/lib/python3.12/site-packages')")
s=s.replace("sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')","")
s=s.replace("sdk=W/'.cache/all-pairs/remote_native_deepgemm_deepep/sdk-920b2'","sdk=Path('/usr/local/Ascend/cann-9.2.0-beta.2')")
(D/'run_case.py').write_text(s)
manifest=json.loads((C/'INSTALLATION_MANIFEST.json').read_text());artifacts=[]
for rel,r in manifest['artifacts'].items():
 f=C/rel;h=hashlib.sha256(f.read_bytes()).hexdigest();assert h==r['sha256'];artifacts.append(dict(path=str(f),sha256=h))
(D/'provenance').mkdir(exist_ok=True);(D/'provenance/0_build_receipt.json').write_text(json.dumps(dict(original=str(C/'INSTALLATION_MANIFEST.json'),artifacts=artifacts,manifest=manifest),indent=2))
for name in ['SOURCE_CHECKS.json','CHECKS.json','SOURCE_OBJECTS.json']:shutil.copyfile(C/name,D/'provenance'/name)
orig=json.loads((P/'original_hashes.json').read_text());mismatch=[]
for rel,h in orig.items():
 f=D/'runtime'/rel
 if f.exists() and hashlib.sha256(f.read_bytes()).hexdigest()!=h:mismatch.append(rel)
assert not mismatch,mismatch
paths=[str(C/'venv/lib/python3.12/site-packages'),str(W/'.venv/lib/python3.12/site-packages'),*sys.path]
deps={n:(spec.origin if (spec:=importlib.machinery.PathFinder.find_spec(n,paths)) else None) for n in ['setuptools','torch','torch_npu','numpy','Cython','psutil','cloudpickle','tqdm']};assert all(deps.values()),deps
(D/'CPU_CHECK.json').write_text(json.dumps(dict(python=sys.version,dependencies=deps,original_frontend_matches=True,compiler_artifacts=artifacts),indent=2));print(json.dumps(deps))
