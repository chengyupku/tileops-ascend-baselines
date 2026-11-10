import io,tarfile,subprocess,hashlib,json
from pathlib import Path
h=Path(__file__).resolve().parent
src=Path('.cache/all-pairs/fla/python_binary_partial/fla_npu')
files=[p for p in src.rglob('*') if p.is_file() and p.suffix in ('.py','.o','.json')]
root='/opt/tileops-baseline-cache/workspace/reports/baseline-retest32-20261030/server1'
ssh=['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495']
subprocess.run(ssh+[f'mkdir -p {root}/deployments/fla_backward/fla_npu'],check=True)
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w') as t:
 for p in files:t.add(p,arcname=str(p.relative_to(src)),recursive=False)
subprocess.run(ssh+[f'tar -xf - -C {root}/deployments/fla_backward/fla_npu'],input=b.getvalue(),check=True)
manifest={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
code='expected='+repr(manifest)+r'''
from pathlib import Path
import hashlib,json,shutil,subprocess,time
h=Path('/opt/tileops-baseline-cache/workspace/reports/baseline-retest32-20261030/server1');dst=h/'deployments/fla_backward/fla_npu';build=h/'native_build/chunk_gated_delta_rule_bwd';v=dst/'opp/vendors/fla_npu_transformer'
for rel,sha in expected.items():assert hashlib.sha256((dst/rel).read_bytes()).hexdigest()==sha
for name,folder in [('libcust_opapi.so','op_api/lib'),('libcust_opsproto_rt2.0.so','op_proto/lib/linux/aarch64'),('libcust_opmaster_rt2.0.so','op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64')]:
 p=v/folder/name;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();shutil.copy2(build/name,p)
p=v/'op_impl/ai_core/tbe/op_tiling/liboptiling.so';p.symlink_to('lib/linux/aarch64/libcust_opmaster_rt2.0.so')
exports=subprocess.check_output(['nm','-D','--defined-only',str(v/'op_api/lib/libcust_opapi.so')],text=True)
for symbol in ('ChunkGatedDeltaRuleBwd','ChunkFwdH','ChunkGatedDeltaRuleFwdPrepare','ChunkGdnBwdIntra','ChunkGatedDeltaRuleBwdDhu','ChunkGatedDeltaRuleBwdFinalize'):assert 'aclnn'+symbol+'GetWorkspaceSize' in exports,symbol
r={'at':time.time(),'original_files':expected,'host_build':str(build),'runtime_verified':False,'package':str(dst)}
(h/'deployments/fla_backward/PROVENANCE.json').write_text(json.dumps(r,indent=2));print(json.dumps({'files':len(expected),'package':str(dst)}))
'''
r=subprocess.run(ssh+['python3 -'],input=code,text=True,capture_output=True,check=True);print(r.stdout)
(h/'BACKWARD_PACKAGE_FILES.json').write_text(json.dumps(manifest,indent=2))
