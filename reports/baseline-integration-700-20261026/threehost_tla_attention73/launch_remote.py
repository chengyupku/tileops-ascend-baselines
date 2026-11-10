import pathlib,subprocess,json,time,os
P=pathlib.Path(__file__).resolve().parent
out=P/'runs/serving-kv512-float16-original-default-attempt02';out.mkdir(parents=True,exist_ok=True)
env=os.environ.copy()
for n in ['TMPDIR','XDG_CACHE_HOME','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR','ASCEND_WORK_PATH','ASCEND_CACHE_PATH','ASCEND_PROCESS_LOG_PATH']:
 d=out/n;d.mkdir(exist_ok=True);env[n]=str(d)
env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MAX_JOBS='2',PYTHONDONTWRITEBYTECODE='1')
cmd=['npu-guard','run','--owner','baseline700-20261026/threehost_tla_attention73','--wait','120','--timeout','240','--',str(P/'private-python'),str(P/'run_case.py'),'MHADecodePagedWithKVCacheFwdOp','serving-kv512-float16']
r={'argv':cmd,'start':time.time()}
(out/'command.json').write_text(json.dumps(r,indent=2))
with (out/'guard.log').open('x') as f:
 q=subprocess.run(cmd,env=env,cwd=out,stdout=f,stderr=subprocess.STDOUT)
r.update(returncode=q.returncode,end=time.time());(out/'guard.json').write_text(json.dumps(r,indent=2))
(out/'guard_metadata_after.json').write_text(pathlib.Path('/run/lock/ascend-npu/device-0.lock').read_text())
print(json.dumps(r))
