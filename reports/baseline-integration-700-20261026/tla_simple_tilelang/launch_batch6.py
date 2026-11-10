from pathlib import Path
import json,sys,subprocess,datetime,time,os
P=Path(__file__).resolve().parent;batch=sys.argv[1];name=Path(batch).stem
scope={(r['operator'],r['case_id']) for r in json.loads((P/'SCOPE.json').read_text()) if r['baseline']=='tilelang-ascend'}
keys=json.loads((P/batch).read_text());assert all(tuple(k) in scope for k in keys)
if (P/'FEEDBACK.txt').exists(): print((P/'FEEDBACK.txt').read_text(),flush=True)
log=P/(name+'.guard.log');assert not log.exists()
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
(P/'HEARTBEAT.json').write_text(json.dumps(dict(utc=now,phase='batch_wait_run',current_keys=keys,last_progress_at=now,blocked_reason=None),indent=2))
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MAX_JOBS='2',PYTHONDONTWRITEBYTECODE='1')
cmd=['npu-guard','run','--owner','baseline700-20261026/tla_simple_tilelang','--wait','3600','--timeout','300','--','/opt/tileops-all-pairs/bin/allpairs-python',str(P/'run_batch6.py'),batch]
record=dict(argv=cmd,started=time.time(),keys=keys)
with log.open('x') as f:r=subprocess.run(cmd,cwd=P,env=env,stdout=f,stderr=subprocess.STDOUT)
record.update(finished=time.time(),returncode=r.returncode)
(P/(name+'.guard.json')).write_text(json.dumps(record,indent=2))
subprocess.run(['python3',str(P/'summarize.py')],check=True)
print(record,flush=True)
raise SystemExit(r.returncode)
