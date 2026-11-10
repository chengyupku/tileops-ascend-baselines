import sys,subprocess,json,time,os,datetime
from pathlib import Path
R=Path(__file__).resolve().parent
for op,cid,attempt in json.loads(Path(sys.argv[1]).read_text()):
 a=R/'runs'/attempt;a.mkdir(parents=True,exist_ok=False)
 t=time.monotonic();start=datetime.datetime.now(datetime.timezone.utc).isoformat()
 cmd=[sys.executable,str(R/'run_case.py'),'sgl-kernel-npu',op,cid,str(a/'output')]
 (a/'COMMAND.json').write_text(json.dumps(cmd))
 with (a/'guard.log').open('x') as f:
  p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 (a/'GUARD_RESULT.json').write_text(json.dumps({'returncode':p.returncode,'started':start,'finished':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-t,'nested_guard':False,'note':'Child stays in parent guard process group', 'batch_plan':sys.argv[1]}))
 print(cid,p.returncode,flush=True)
