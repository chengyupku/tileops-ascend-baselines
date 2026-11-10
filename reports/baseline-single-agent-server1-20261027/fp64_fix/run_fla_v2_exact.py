from pathlib import Path
import subprocess,shlex,time,json
P=Path(__file__).resolve().parent;W=P.parents[2];R='/opt/tileops-all-pairs/workspace/'+str(P.relative_to(W));ssh=str(W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh');host='openlibing-DevEnv-661495'
def call(label,cmd):
 start=time.time()
 with (P/(label+'.guard.log')).open('w') as f:a=subprocess.run([ssh,host,shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
 (P/(label+'.guard.json')).write_text(json.dumps({'command':cmd,'exit_code':a.returncode,'start':start,'end':time.time()},indent=2));print(label,a.returncode,flush=True);return a.returncode
assert call('fla-v2-exact-build',['npu-guard','run','--owner','single-server1/fp64','--wait','1200','--timeout','1800','--','bash','-c','source /usr/local/Ascend/ascend-toolkit/set_env.sh && python3 '+shlex.quote(R+'/compile_v2_exact.py')])==0
code=f'''from pathlib import Path
p=Path({R!r})
s=(p/'run_case_fla.py').read_text().replace("P/'runs_fla_deploy_retry'","P/'runs_fla_binary_retry'").replace("str(P/'checker.py')","str(T/'scripts/ascend_pair_check.py')")
s=s.replace("'--output',str(out/'checker.json')","'--output',str(out/'checker.json'),'--cpu-fp64-reference'")
(p/'run_case_fla_binary.py').write_text(s)
'''
subprocess.run([ssh,host,'python3 -'],input=code,text=True,check=True)
call('fla-kda-binary-retry',['npu-guard','run','--owner','single-server1/fp64','--wait','1200','--timeout','1800','--','env','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','LD_LIBRARY_PATH='+R+'/fla_package/fla_npu/opp/vendors/fla_npu_transformer/op_api/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/driver/lib64/common','/opt/tileops-all-pairs/bin/allpairs-python',R+'/run_case_fla_binary.py','flash-linear-attention-npu','KDAFwdOp','kda-tp1-16x8192-bfloat16'])
