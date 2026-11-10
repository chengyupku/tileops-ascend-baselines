from pathlib import Path
import subprocess,json,os,time,sys
P=Path(__file__).resolve().parent;op,cid=sys.argv[1:];out=P/'local_runs'/(cid+'-pto-A5-original-branch');out.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
for n in ['TMPDIR','XDG_CACHE_HOME','ASCEND_WORK_PATH','ASCEND_CACHE_PATH','ASCEND_PROCESS_LOG_PATH']:
 d=out/n;d.mkdir(exist_ok=True);env[n]=str(d)
cmd=['npu-guard','run','--owner','baseline700-20261026/threehost_attention_mla_mha31','--wait','120','--timeout','180','--',str(P/'local-python'),str(P/'run_local.py'),op,cid]
r=dict(argv=cmd,start=time.time())
(out/'command.json').write_text(json.dumps(r,indent=2))
with (out/'guard.log').open('x') as f:r['returncode']=subprocess.run(cmd,env=env,cwd=out,stdout=f,stderr=subprocess.STDOUT).returncode
r['end']=time.time();(out/'guard.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
