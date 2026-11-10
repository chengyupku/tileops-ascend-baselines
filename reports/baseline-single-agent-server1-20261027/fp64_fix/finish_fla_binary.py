from pathlib import Path
import subprocess,json,shlex,time,tarfile,io
P=Path(__file__).resolve().parent;W=P.parents[2];R='/opt/tileops-all-pairs/workspace/'+str(P.relative_to(W))
assert json.loads((P/'flash-linear-attention-npu.binary-build.json').read_text())['exit_code']==0
ssh=str(W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh');host='openlibing-DevEnv-661495'
code='''from pathlib import Path
import shutil,json,hashlib
p=Path(ROOT)
src=p/'original-build/flash-linear-attention-npu/build/_CPack_Packages/Linux/External/fla_npu_linux-aarch64.run/packages/vendors/fla_npu_transformer'
assert src.is_dir()
files=list(src.rglob('*.o'));assert files, 'Original package lacks binary kernels'
dst=p/'fla_package/fla_npu/opp/vendors/fla_npu_transformer'
shutil.copytree(dst,p/'fla_jit_vendor_before',dirs_exist_ok=False)
shutil.copytree(src,dst,dirs_exist_ok=True)
(p/'FLA_BINARY_DEPLOYMENT.json').write_text(json.dumps({'binary_objects':len(files),'source':str(src),'destination':str(dst),'hashes':{str(f.relative_to(dst)):hashlib.sha256(f.read_bytes()).hexdigest() for f in dst.rglob('*') if f.is_file()}},indent=2))
s=(p/'run_case_fla.py').read_text().replace("P/'runs_fla_deploy_retry'","P/'runs_fla_binary_retry'").replace("str(P/'checker.py')","str(T/'scripts/ascend_pair_check.py')")
s=s.replace("'--output',str(out/'checker.json')","'--output',str(out/'checker.json'),'--cpu-fp64-reference'")
(p/'run_case_fla_binary.py').write_text(s)
print('Original binary objects installed',len(files))
'''.replace('ROOT',repr(R))
a=subprocess.run([ssh,host,'python3 -'],input=code,text=True,check=True)
cmd=['npu-guard','run','--owner','single-server1/fp64','--wait','1200','--timeout','1800','--','env','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','LD_LIBRARY_PATH='+R+'/fla_package/fla_npu/opp/vendors/fla_npu_transformer/op_api/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/driver/lib64/common','/opt/tileops-all-pairs/bin/allpairs-python',R+'/run_case_fla_binary.py','flash-linear-attention-npu','KDAFwdOp','kda-tp1-16x8192-bfloat16']
t=time.time()
with (P/'fla-kda-binary-retry.guard.log').open('w') as f:a=subprocess.run([ssh,host,shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
(P/'fla-kda-binary-retry.guard.json').write_text(json.dumps({'command':cmd,'exit_code':a.returncode,'start':t,'end':time.time()},indent=2));print('FLA full binary checker exit',a.returncode)
