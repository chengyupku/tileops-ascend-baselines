import sys,os,json,platform
from pathlib import Path
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O))
import registration as M
pins=M.metadata()['PINS.json']
if platform.machine()!=pins['host']['machine'] or platform.node()!=pins['host']['node']:raise RuntimeError('Tested server1 only')
M.require_guard()
os.environ.update(pins['environment'])
os.environ.update(TORCH_DEVICE_BACKEND_AUTOLOAD='0',FLA_NPU_DISABLE_PTH='1',TRITON_BACKEND='torch_npu',ASCEND_GLOBAL_LOG_LEVEL='0',ATB_HOME_PATH='/usr/local/Ascend/nnal/atb/9.1.0/atb/cxx_abi_1',ASCEND_AICPU_PATH='/usr/local/Ascend/cann-9.1.0')
root=Path(os.environ['BMM_VARIANT_RUN_ROOT'])
if root.parent!=M.REMOTE_ROOT/'runs' or not (root/'RUN_EXCLUSIVE').is_file():raise RuntimeError('Private run required')
for n in M.CACHE_NAMES:
 p=root/n;p.mkdir();os.environ[n]=str(p)
# Preflight runtime bytes before startup helper/backend code; all execution remains
# in the same foreground process group owned by npu-guard.
for p,h in pins['runtime'].items():M.checked(p,h)
exe='/opt/tileops-ci/runtime/bin/python'
os.execv(exe,[exe,'-I','-S','-B','/opt/tileops-all-pairs/control/private_python.py',str(O/'evaluate.py')])
