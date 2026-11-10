"""Deployment recipe only; execute explicitly after parent review. No automatic checker run."""
import sys,os,json,tempfile,subprocess
from pathlib import Path
sys.dont_write_bytecode=True
O=Path(__file__).resolve().parent
import registration as M
if len(sys.argv)<2:raise SystemExit('Usage: run_environment.py <foreground Python script> [args]; never run this for registration-only review')
subprocess.run([sys.executable,'-B',str(O/'preflight.py')],check=True)
root=O/'runs';root.mkdir(exist_ok=True);run=Path(tempfile.mkdtemp(prefix='run-',dir=root));(run/'RUN_EXCLUSIVE').touch(exist_ok=False)
env=os.environ.copy();cfg=M.metadata()['FROZEN.json']['groups']['attention'];env.update(cfg['environment']);env['LD_LIBRARY_PATH']=cfg['ld_library_path_prefix'];env['ASCEND_CANN_PACKAGE_PATH']='/usr/local/Ascend/cann-9.1.0'
env['PATH']='/usr/local/Ascend/cann-9.1.0/tools/bisheng_compiler/bin:/usr/local/Ascend/cann-9.1.0/compiler/ccec_compiler/bin:'+env.get('PATH','')
owner='baseline700-20261026/threehost_dsa_registration2';env['DSA2_GUARD_OWNER']=owner;env['DSA2_RUN_ROOT']=str(run)
for name in M.CACHE_NAMES:(run/name).mkdir();env[name]=str(run/name)
for name in ('TMPDIR','TMP','TEMP'):env[name]='/proc/self/cwd/'+name
# All original inherited OMP/MKL/OpenBLAS settings remain untouched.
cmd=['npu-guard','run','--owner',owner,'--wait','600','--timeout','1200','--','/opt/tileops-ci/runtime/bin/python','-I','-S','-B','/opt/tileops-all-pairs/control/private_python.py',*sys.argv[1:]]
(run/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (run/'guard.log').open('w') as f:r=subprocess.run(cmd,env=env,cwd=run,stdout=f,stderr=subprocess.STDOUT)
(run/'guard.json').write_bytes(Path('/run/lock/ascend-npu/device-0.lock').read_bytes())
raise SystemExit(r.returncode)
