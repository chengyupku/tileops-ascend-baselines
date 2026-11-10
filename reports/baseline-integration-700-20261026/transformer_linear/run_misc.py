from pathlib import Path
import json,sys,subprocess,time,datetime
W=Path(__file__).resolve().parent;i=sys.argv[1];d=W/i;assert not (d/'GUARD_RECEIPT.json').exists();t=time.time()
argv=['npu-guard','run','--owner','baseline700-20261026/transformer_linear','--wait','3600','--timeout','300','--','bash',str(d/'child.sh')]
(W/'HEARTBEAT.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase='misc_guard',current_keys=[i],last_progress_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),blocked_reason=None)))
with (d/'guard.stdout.log').open('w') as out,(d/'guard.stderr.log').open('w') as err:p=subprocess.run(argv,stdout=out,stderr=err)
(d/'GUARD_RECEIPT.json').write_text(json.dumps(dict(argv=argv,start=t,end=time.time(),returncode=p.returncode,elapsed_seconds=time.time()-t),indent=2));print(i,p.returncode)
