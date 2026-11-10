from pathlib import Path
import subprocess,tarfile,io,json,sys,shlex,time,hashlib
P=Path(__file__).resolve().parent;W=P.parents[2];R='/opt/tileops-all-pairs/workspace/'+str(P.relative_to(W))
ssh=str(W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh');host='openlibing-DevEnv-661495'
if sys.argv[1]=='push':
 b=io.BytesIO()
 with tarfile.open(fileobj=b,mode='w') as t:
  for p in P.glob('*.py'):t.add(p,arcname=p.name)
  for p in P.glob('*.json'):t.add(p,arcname=p.name)
  for p in (P/'vllm_private').glob('*.py'):t.add(p,arcname='vllm_private/'+p.name)
 subprocess.run([ssh,host,'mkdir -p '+shlex.quote(R)],check=True)
 subprocess.run([ssh,host,'tar -xf - -C '+shlex.quote(R)],input=b.getvalue(),check=True)
elif sys.argv[1]=='sgl':
 src='/opt/tileops-all-pairs/build/sgl-pristine-20261025T2310/source/python/sgl_kernel_npu'
 a=subprocess.run([ssh,'openlibing-DevEnv-055409',f'tar --exclude=__pycache__ -cf - -C {src} sgl_kernel_npu'],stdout=subprocess.PIPE,check=True)
 (P/'sgl-deployment.tar').write_bytes(a.stdout)
 subprocess.run([ssh,host,'mkdir -p '+src],check=True)
 subprocess.run([ssh,host,f'tar -xf - -C {src}'],input=a.stdout,check=True)
 (P/'SGL_DEPLOYMENT.json').write_text(json.dumps({'source_host':'openlibing-DevEnv-055409','destination_host':host,'archive_sha256':hashlib.sha256(a.stdout).hexdigest(),'bytes':len(a.stdout)},indent=2))
elif sys.argv[1] in ('run','retry'):
 lib,op,cid=sys.argv[2:];label=lib+'--'+cid+('--deps-retry' if sys.argv[1]=='retry' else '')
 cmd=['npu-guard','run','--owner','single-server1/fp64','--wait','600','--timeout','1800','--','env','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','LD_LIBRARY_PATH=/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_host_builds/toolchain/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64','/opt/tileops-all-pairs/bin/allpairs-python',R+('/run_case_retry.py' if sys.argv[1]=='retry' else '/run_case.py'),lib,op,cid]
 start=time.time()
 with (P/(label+'.guard.log')).open('w') as f:r=subprocess.run([ssh,host,shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
 (P/(label+'.guard.json')).write_text(json.dumps({'command':cmd,'exit_code':r.returncode,'start':start,'end':time.time()},indent=2));print(label,r.returncode,flush=True)
elif sys.argv[1]=='pull':
 cmd=['tar','--exclude=TMPDIR','--exclude=ASCEND_CACHE_PATH','--exclude=ASCEND_WORK_PATH','--exclude=ASCEND_PROCESS_LOG_PATH','--exclude=TRITON_CACHE_DIR','--exclude=XDG_CACHE_HOME','--exclude=TORCH_EXTENSIONS_DIR','-cf','-','-C',R,'runs']
 r=subprocess.run([ssh,host,shlex.join(cmd)],stdout=subprocess.PIPE,check=True)
 with tarfile.open(fileobj=io.BytesIO(r.stdout)) as t:t.extractall(P,filter='data')
