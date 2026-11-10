from pathlib import Path
import sys,json,os,subprocess
R=Path(__file__).resolve().parent
op=sys.argv[1]; cid=sys.argv[2] if len(sys.argv)>2 else None
rows=json.loads((R/'SCOPE.json').read_text()); row=next(x for x in rows if x['operator']==op and (cid is None or x['case_id']==cid))
case=R/'runs'/f"{op}--{row['case_id']}"; case.mkdir(parents=True,exist_ok=False)
cmd=['npu-guard','run','--owner','baseline700-20261026/sgl_gdn_recovery02','--wait','3600','--timeout','300','--','/root/dyq/workspace/.tools/baselines/baseline-run',row['baseline'],'python',str(R/'run_case.py'),row['baseline'],row['operator'],row['case_id'],str(case/'output')]
(case/'COMMAND.json').write_text(json.dumps(cmd,indent=2)); env=os.environ.copy(); env['PYTHONDONTWRITEBYTECODE']='1'; env['FLA_NPU_DISABLE_PTH']='1'
with (case/'guard.log').open('x') as f: p=subprocess.run(cmd,cwd=case,env=env,stdout=f,stderr=subprocess.STDOUT)
(case/'GUARD_RESULT.json').write_text(json.dumps({'returncode':p.returncode,'automatic_retry':False})); print(op,row['case_id'],p.returncode)
