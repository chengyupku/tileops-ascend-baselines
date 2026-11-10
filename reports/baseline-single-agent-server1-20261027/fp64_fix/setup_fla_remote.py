from pathlib import Path
import shutil,json,hashlib,subprocess,os
P=Path(__file__).resolve().parent;W=Path('/opt/tileops-all-pairs/workspace')
source=P/'original-build/flash-linear-attention-npu'
package=P/'fla_package/fla_npu'
shutil.copytree(source/'torch_custom/fla_npu/fla_npu',package,dirs_exist_ok=True)
vendor=source/'build/_CPack_Packages/Linux/External/fla_npu_linux-aarch64.run/packages/vendors/fla_npu_transformer'
assert vendor.is_dir()
shutil.copytree(vendor,package/'opp/vendors/fla_npu_transformer',dirs_exist_ok=True)
original=W/'ascend_baselines/flash-linear-attention-npu'
assert not original.exists();original.symlink_to(source,target_is_directory=True)
lib=package/'opp/vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so'
r=subprocess.run(['nm','-D',str(lib)],capture_output=True,text=True,check=True)
assert ' aclnnChunkKdaFwd\n' in r.stdout and ' aclnnChunkKdaFwdGetWorkspaceSize\n' in r.stdout
config=package/'opp/vendors/fla_npu_transformer/op_impl/ai_core/tbe/config/ascend950/aic-ascend950-ops-info.json'
assert 'ChunkKdaFwd' in json.loads(config.read_text())
(P/'FLA_DEPLOYMENT.json').write_text(json.dumps({'package':str(package),'source':str(source),'api_hash':hashlib.sha256(lib.read_bytes()).hexdigest(),'elf_machine':lib.read_bytes()[18:20].hex(),'config_ops':list(json.loads(config.read_text())),'original_build':True},indent=2))
print('fla deployed',lib.stat().st_size)
