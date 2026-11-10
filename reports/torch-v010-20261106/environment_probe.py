import pathlib,json,hashlib,platform,os,subprocess,importlib.metadata as md
root=pathlib.Path('/usr/local/Ascend/cann-9.1.0');site=pathlib.Path('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
files={};versions={}
for p in root.rglob('*version*'):
 if p.is_file() and p.stat().st_size<65536 and p.suffix in ('.info','.cfg','.json','.txt'):versions[str(p.relative_to(root))]=p.read_text(errors='replace')
for base,pattern in [(root,'libascendcl.so'),(root,'libopapi.so'),(root,'libnnopbase.so'),(root,'libhccl.so'),(root,'libprofapi.so'),(root,'libmsprofiler.so'),(root,'bisheng'),(pathlib.Path('/usr/local/Ascend/driver'),'libascend_hal.so'),(site/'torch/lib','libtorch_cpu.so'),(site/'torch_npu/lib','libtorch_npu.so')]:
 for p in base.rglob(pattern):
  if p.is_file():files[str(p)]=sha(p)
dists={d.metadata['Name']:d.version for d in md.distributions(path=[str(site)])if d.metadata.get('Name')in ['torch','torch-npu','numpy','tilelang','triton-ascend']}
print(json.dumps({'host':platform.node(),'arch':platform.machine(),'kernel':platform.release(),'packages':dists,'versions':versions,'binary_sha256':files,'driver_version':pathlib.Path('/usr/local/Ascend/driver/version.info').read_text(),'guard_sha256':sha(pathlib.Path('/usr/local/libexec/ascend-npu/npu_guard.py')),'free_bytes':os.statvfs('/opt').f_bavail*os.statvfs('/opt').f_frsize},indent=2))
