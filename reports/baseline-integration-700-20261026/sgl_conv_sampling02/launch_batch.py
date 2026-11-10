from pathlib import Path
import subprocess,json,os,time,datetime,sys
R=Path(__file__).resolve().parent;a=R/sys.argv[1];a.mkdir(exist_ok=False)
cmd=[str(R/'RUN_GUARDED.sh'),str(R/'batch_device.py'),str(R/sys.argv[2])]
(a/'COMMAND.json').write_text(json.dumps(cmd));t=time.monotonic();start=datetime.datetime.now(datetime.timezone.utc).isoformat()
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
with (a/'guard.log').open('x') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
(a/'GUARD_RESULT.json').write_text(json.dumps({'returncode':p.returncode,'started':start,'finished':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-t}))
print(p.returncode)
