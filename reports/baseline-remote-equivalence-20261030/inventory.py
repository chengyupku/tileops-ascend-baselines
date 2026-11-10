import sys,json,socket,hashlib,ctypes as C,time,subprocess
from pathlib import Path
root=Path(sys.argv[1]);hal=C.CDLL('/usr/local/Ascend/driver/lib64/driver/libascend_hal.so');hw={}
for module,label in [(4,'aicore'),(7,'vectorcore')]:
 v=C.c_int64();rc=hal.halGetDeviceInfo(0,module,3,C.byref(v));hw[label]={'rc':rc,'count':v.value}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
vendor=root/'reports/baseline-old1076-20261029/arm_nn_complete_subset/packages/vendors/allpairs_nn'
files={str(p.relative_to(vendor)):sha(p) for p in vendor.rglob('*') if p.is_file()}
sources={p:sha(root/p) for p in ['TileOPs/scripts/cache_frozen_pair_check.py','tools/measure_baselines.py','tools/baseline_protocol.py','tools/baseline_old1076_nn_arm_staging_v3.py']}
adapters=root/'reports/baseline-old1076-20261029/arm_nn_adapters'
source_adapters={str(p.relative_to(adapters)):sha(p) for p in adapters.rglob('*.py')}
libs={}
for path in ['/usr/local/Ascend/driver/lib64/driver/libascend_hal.so','/usr/local/Ascend/cann-9.1.0/lib64/libascendcl.so','/usr/local/Ascend/cann-9.1.0/lib64/libopapi.so']:
 p=Path(path)
 if p.is_file():libs[path]=sha(p)
versions={}
for p in Path('/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages').glob('torch*.dist-info/METADATA'):
 versions[str(p.parent.name)]=[l for l in p.read_text().splitlines() if l.startswith(('Name:','Version:'))]
print(json.dumps({'at':time.time(),'hostname':socket.gethostname(),'hardware':hw,'driver_version':Path('/usr/local/Ascend/driver/version.info').read_text(),'sources':sources,'adapter_hashes':source_adapters,'vendor_hashes':files,'runtime_libs':libs,'package_versions':versions,'guard':json.loads(subprocess.run(['npu-guard','status','--json'],capture_output=True,text=True).stdout)},ensure_ascii=False))
