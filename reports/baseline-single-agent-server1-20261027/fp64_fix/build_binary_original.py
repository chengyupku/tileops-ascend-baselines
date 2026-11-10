from pathlib import Path
import subprocess,shlex,time,json,sys
P=Path(__file__).resolve().parent; W=P.parents[2]; lib=sys.argv[1]
assert lib in ['ops-nn','flash-linear-attention-npu']
R='/opt/tileops-all-pairs/workspace/'+str(P.relative_to(W))
ops='add_layer_norm' if lib=='ops-nn' else 'chunk_kda_fwd'
vendor='single_s1_nn' if lib=='ops-nn' else 'fla_npu_transformer'
args=['env','PYTHONPATH='+R+'/build_python_deps','PIP_NO_INDEX=1','PIP_FIND_LINKS='+R+'/build_wheels','ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest','ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest','ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest','LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/driver/lib64/common','TORCH_DEVICE_BACKEND_AUTOLOAD=0','bash','build.sh','--noaicpu','--soc=ascend950','--ops='+ops,'--vendor_name='+vendor,'-j4']
cmd='cd '+shlex.quote(R+'/original-build/'+lib)+' && '+shlex.join(args)
t=time.time()
with (P/(lib+'.binary-build.log')).open('w') as f:r=subprocess.run([str(W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'),'openlibing-DevEnv-661495',cmd],stdout=f,stderr=subprocess.STDOUT)
(P/(lib+'.binary-build.json')).write_text(json.dumps({'cmd':cmd,'exit_code':r.returncode,'start':t,'end':time.time(),'original_offline_binary_package_build':True},indent=2));print(lib,r.returncode)
