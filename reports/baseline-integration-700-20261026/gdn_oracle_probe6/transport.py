from pathlib import Path
import subprocess,sys,tarfile,io,json,time,shlex
P=Path(__file__).resolve().parent
ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';host='openlibing-DevEnv-055409';remote='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/gdn_oracle_probe6'
if sys.argv[1]=='push':
 b=io.BytesIO()
 with tarfile.open(fileobj=b,mode='w') as t:
  for p in P.iterdir():
   if p.is_file():t.add(p,arcname=p.name)
 r=subprocess.run([ssh,host,f'mkdir -p {remote} && tar -xf - -C {remote} && python3 {remote}/remote_setup.py before'],input=b.getvalue());sys.exit(r.returncode)
if sys.argv[1]=='run':
 assert not (P/'GUARD_REQUEST.json').exists()
 cmd=['npu-guard','run','--owner','baseline700-20261026/gdn_oracle_probe6','--wait','3600','--timeout','300','--','env','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','NUMEXPR_NUM_THREADS=4','VECLIB_MAXIMUM_THREADS=4','PYTHONDONTWRITEBYTECODE=1','bash',remote+'/baseline-run','sgl-kernel-npu','python',remote+'/run_case.py','GDNFwdOp','prefill-narrow-state-bfloat16','probe01']
 start=time.time();(P/'GUARD_REQUEST.json').write_text(json.dumps(dict(command=cmd,start_epoch=start),indent=2))
 with (P/'guard.log').open('w') as f:r=subprocess.run([ssh,host,shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
 (P/'GUARD_RESULT.json').write_text(json.dumps(dict(command=cmd,exit_code=r.returncode,start_epoch=start,end_epoch=time.time()),indent=2));sys.exit(r.returncode)
if sys.argv[1]=='pull':
 subprocess.run([ssh,host,f'python3 {remote}/remote_setup.py after'],check=True)
 r=subprocess.run([ssh,host,f'tar --exclude="*/ASCEND_PROCESS_LOG_PATH" --exclude="*/ASCEND_CACHE_PATH" --exclude="*/tmp" --exclude="*/XDG_CACHE_HOME" --exclude="*/TRITON_CACHE_DIR" --exclude="*/TORCH_EXTENSIONS_DIR" --exclude="*/ASCEND_WORK_PATH" -cf - -C {remote} runs SHARED_HASHES_before.json SHARED_HASHES_after.json SHARED_UNCHANGED.json REMOTE_PREFLIGHT.json'],stdout=subprocess.PIPE,check=True)
 with tarfile.open(fileobj=io.BytesIO(r.stdout)) as t:t.extractall(P,filter='data')
