"""CPU-only fresh run preparation. Never starts a guard or imports runtime libraries."""
import json,os,pathlib,tempfile,sys,shlex
import registration as r
root=pathlib.Path(__file__).resolve().parent
if root!=r.REMOTE_ROOT:raise SystemExit('Stage private package on server1 first')
pins=r.metadata()['PINS.json']
(root/'runs').mkdir(exist_ok=True)
run=pathlib.Path(tempfile.mkdtemp(prefix='run-',dir=root/'runs'))
(run/'RUN_EXCLUSIVE').write_text('fresh run; never reuse\n')
env=dict(pins['environment'],TILELANG3_RUN_ROOT=str(run),TILELANG3_GUARD_OWNER='baseline700/tilelang-registration3',PYTHONDONTWRITEBYTECODE='1')
for name in r.CACHE_NAMES:
 (run/name).mkdir();env[name]=str(run/name)
lines=['unset LD_PRELOAD PYTHONHOME TILELANG_LIBRARY_PATH TVM_LIBRARY_PATH TVM_HOME TVM_IMPORT_PYTHON_PATH TL_ROOT TL_PLATFORM ASCEND_NPU_ARCH']
lines+=['export '+k+'='+shlex.quote(v) for k,v in env.items()]
(run/'environment.sh').write_text('\n'.join(lines)+'\n')
print(run)
