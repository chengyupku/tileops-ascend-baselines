from pathlib import Path
import sys,json,os,subprocess
R=Path(__file__).resolve().parent
op=sys.argv[1];row=next(x for x in json.loads((R.parent/'SCOPE.json').read_text()) if x['operator']==op)
case=R/'runs'/op;case.mkdir(parents=True,exist_ok=False)
cmd=['npu-guard','run','--owner','selected-integration-20261026/nn_misc_math_recovery01','--wait','3600','--timeout','300','--','/root/dyq/workspace/.tools/baselines/baseline-run',row['baseline'],'python',str(R/'run_case.py'),*row['key'],str(case/'output')]
(case/'COMMAND.json').write_text(json.dumps(cmd,indent=2))
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['FLA_NPU_DISABLE_PTH']='1'
with (case/'guard.log').open('x') as f:p=subprocess.run(cmd,cwd=case,env=env,stdout=f,stderr=subprocess.STDOUT)
(case/'GUARD_RESULT.json').write_text(json.dumps({'returncode':p.returncode,'automatic_retry':False}))
print(op,p.returncode)
