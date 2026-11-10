"""CPU-only deployment preflight. Never imports torch or invokes native/compiler binaries."""
import sys,os,json,hashlib,platform,shutil
from pathlib import Path
sys.dont_write_bytecode=True
O=Path(__file__).resolve().parent
import registration as M
assert platform.machine()=='aarch64' and platform.node()=='897b29c557a5','Original server2 required'
data=M.snapshot(runtime=True)
cfg=M.metadata()['FROZEN.json']['groups']['attention']
for p in cfg['directories']:assert Path(p).is_dir(),p
for p in cfg['python_paths']:assert Path(p).is_dir(),p
for exe,p in [('bisheng','/usr/local/Ascend/cann-9.1.0/tools/bisheng_compiler/bin/bisheng')]:assert Path(p).is_file() and os.access(p,os.X_OK),p
for pkg in ('torch','torch_npu'):
 assert Path('/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages',pkg,'__init__.py').is_file()
assert len('/proc/self/cwd/TMPDIR')<60
print(json.dumps({'status':'passed','runtime_pins':len(M.metadata()['PINS.json']['runtime']),'source_files':len(data),'device_imports':0,'compiler_invocations':0}))
