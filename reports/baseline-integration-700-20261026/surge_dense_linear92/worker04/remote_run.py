from pathlib import Path
import os,sys,json,hashlib,subprocess,datetime,time
W=Path(__file__).resolve().parent;P=W/'package';case=sys.argv[1];A=W/sys.argv[2];A.mkdir()
os.chdir(W)
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
r=json.loads((P/'RUNTIME_RECIPE.json').read_text());env=os.environ.copy();env.update(r['required_environment']);env['LD_LIBRARY_PATH']=':'.join(r['ld_library_path_order']);env['PYTHONPATH']=':'.join(r['python_paths']);env['CAPTURE_DIR']=str(A)
env.update(ASCEND_GLOBAL_LOG_LEVEL='0',ASCEND_SLOG_PRINT_TO_STDOUT='1',ASCEND_PROCESS_LOG_PATH=str(A/'native_logs'),TMPDIR=str(A/'tmp'),TORCH_EXTENSIONS_DIR=str(A/'torch_extensions'),XDG_CACHE_HOME=str(A/'cache'))
for n in ['tmp','cache','native_logs','torch_extensions']:(A/n).mkdir()
pins=json.loads((P/'PINS.json').read_text());frozen={}
for row in pins:
 p=Path(row['path']);actual=hashlib.sha256(p.read_bytes()).hexdigest();assert actual==row['sha256'],str(p);frozen[str(p)]={'resolved':str(p.resolve()),'sha256':actual}
paths=[Path('/opt/tileops-ci/runtime/bin/python').resolve(),Path('/usr/local/bin/npu-guard')]+list(Path(r['official_source']).glob('**/libtiling*.so'))+list(P.glob('*'))+[W/'ALL_CONTRACTS.json',Path(__file__),Path(r['checker'])]
root=Path("/opt/tileops-all-pairs/workspace")
paths+=list((root/'TileOPs/workloads').glob('numerics.py'))+list((root/'TileOPs/workloads/linear_attention').glob('gdn.py'))
for p in paths:
 if p.is_file():frozen[str(p)]={'resolved':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(A/'FROZEN.json').write_text(json.dumps(dict(utc=now(),files=frozen,environment={k:env[k] for k in list(r['required_environment'])+['LD_LIBRARY_PATH','PYTHONPATH','CAPTURE_DIR','ASCEND_GLOBAL_LOG_LEVEL','ASCEND_SLOG_PRINT_TO_STDOUT','ASCEND_PROCESS_LOG_PATH','TMPDIR','TORCH_EXTENSIONS_DIR','XDG_CACHE_HOME']}),indent=2))
cmd=['npu-guard','run','--owner',r['guard']['owner'],'--wait','120','--timeout','600','--','/opt/tileops-ci/runtime/bin/python',str(P/'entry.py'),'--op','GDNFwdOp','--case-id',case,'--library','ops-transformer','--output',str(A/'checker.json')]
(A/'STARTED.json').write_text(json.dumps(dict(utc=now(),case=case,command=cmd,api_queue_seconds=None),indent=2))
with (A/'raw.log').open('w') as log:
 proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 while True:
  (W/'HEARTBEAT.json').write_text(json.dumps(dict(utc=now(),attempt=A.name,case=case,phase='guard_or_execution',pid=proc.pid)))
  try:rc=proc.wait(timeout=30);break
  except subprocess.TimeoutExpired:pass
(A/'LOCK_FINAL.json').write_text(Path('/run/lock/ascend-npu/device-0.lock').read_text())
(A/'GUARD_RESULT.json').write_text(json.dumps(dict(utc=now(),returncode=rc,api_queue_seconds=None),indent=2));print(json.dumps(dict(case=case,returncode=rc)),flush=True)
