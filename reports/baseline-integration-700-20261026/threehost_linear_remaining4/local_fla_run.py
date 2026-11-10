import os,sys,json,subprocess,time,datetime,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;cid=sys.argv[1];D=P/('fla_'+cid+('_'+sys.argv[2] if len(sys.argv)>2 else ''));D.mkdir(exist_ok=False)
# Canonical input fixture/checker and candidate are unchanged. All caches are private.
package='/root/dyq/workspace/.cache/ascend-classic/native-pilot-20261020/flash-linear-attention-npu/python_binary/fla_npu'
script='''set -e
source /usr/local/Ascend/cann-9.2.0-beta.2/set_env.sh
export FLA_NPU_DISABLE_PTH=1 PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="/root/dyq/workspace/.cache/ascend-classic/native-pilot-20261020/flash-linear-attention-npu/python_binary"
export ASCEND_CLASSIC_FLA_PACKAGE="'''+package+'''"
export ASCEND_CUSTOM_OPP_PATH="'''+package+'''/opp/vendors/fla_npu_transformer:'''+package+'''/opp"
export LD_LIBRARY_PATH="'''+package+'''/opp/vendors/fla_npu_transformer/op_api/lib:${LD_LIBRARY_PATH:-}"
exec /root/dyq/workspace/.venv/bin/python -B "'''+str(P/'local_fla_entry.py')+'" "'+cid+'" "'+str(D)+'"\n'
(D/'child.sh').write_text(script);env=os.environ.copy();env.update(ASCEND_GLOBAL_LOG_LEVEL='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2');env.pop('ASCEND_NPU_ARCH',None)
for n in ['TMPDIR','XDG_CACHE_HOME','ASCEND_PROCESS_LOG_PATH','ASCEND_WORK_PATH','ASCEND_CACHE_PATH','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR']:
 (D/n).mkdir();env[n]=str(D/n)
cmd=['npu-guard','run','--owner','baseline700-20261026/threehost_linear_remaining4','--wait','120','--timeout','300','--','bash',str(D/'child.sh')]
(D/'FROZEN.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [P/'local_fla_entry.py',P/'integration700_fla_native.py',P/'original_fla_gdn_checker.py',P/'SNAPSHOT_HASHES.json']},indent=2))
t=time.time()
with (D/'stdout.log').open('w') as o,(D/'stderr.log').open('w') as e:r=subprocess.run(cmd,env=env,stdout=o,stderr=e)
(D/'GUARD_RESULT.json').write_text(json.dumps({'argv':cmd,'returncode':r.returncode,'wall_seconds':time.time()-t,'metadata':json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())},indent=2));print(cid,r.returncode)
