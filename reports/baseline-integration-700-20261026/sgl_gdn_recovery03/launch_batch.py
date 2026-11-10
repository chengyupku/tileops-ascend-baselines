from pathlib import Path
import sys,json,os,subprocess,time,datetime
R=Path(__file__).resolve().parent; batch=Path(sys.argv[1]).resolve()
name=batch.stem; root=R/'batches'/name; root.mkdir(parents=True,exist_ok=False)
if (R/'FEEDBACK.txt').exists(): print((R/'FEEDBACK.txt').read_text(),flush=True)
cmd=['npu-guard','run','--owner','baseline700-20261026/sgl_gdn_recovery03','--wait','3600','--timeout','300','--','/root/dyq/workspace/.tools/baselines/baseline-run','sgl-kernel-npu','python',str(R/'batch.py'),str(batch)]
(root/'COMMAND.json').write_text(json.dumps(cmd,indent=2)); env=os.environ.copy(); env['PYTHONDONTWRITEBYTECODE']='1'; env['FLA_NPU_DISABLE_PTH']='1'
start=time.monotonic(); at=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (root/'guard.log').open('x') as f: p=subprocess.run(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
(root/'GUARD_RESULT.json').write_text(json.dumps({'returncode':p.returncode,'started_at':at,'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start,'automatic_retry':False})); print(name,p.returncode)
