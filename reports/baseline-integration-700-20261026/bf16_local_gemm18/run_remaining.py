from pathlib import Path
import json,subprocess,sys,yaml,datetime
T=Path(__file__).resolve().parent;W=Path('/root/dyq/workspace');scope=json.loads((T/'SCOPE.json').read_text());keys={x['case_id'] for x in scope}
rows=yaml.safe_load((W/'TileOPs/src/tileops/manifest/spec/gemm.yaml').read_text())['GemmFwdOp']['workloads'];rows=[dict(x,case_id=x['label']+'-bfloat16') for x in rows if x['label']+'-bfloat16' in keys]
rows.sort(key=lambda x:(x['M']*x['K']+x['K']*x['N']+x['M']*x['N'],x['M']*x['N']*x['K']))
assert len(rows)==18
(T/'EXECUTION_ORDER.json').write_text(json.dumps(rows,indent=2))
for row in rows:
 key=row['case_id']
 if key=='square-1k-bfloat16':continue
 p=T/key
 if p.exists():raise RuntimeError('No rerun '+key)
 subprocess.run([sys.executable,str(T/'prepare_case.py'),key],check=True)
 rc=subprocess.run([sys.executable,str(T/'run_stage.py'),key,'300',str(p/'run.sh')]).returncode
 if rc==0:
  subprocess.run([sys.executable,str(T/'validate_case.py'),str(p)],check=True)
 elif rc in [75,76,78]:
  (T/'BATCH_STOP.json').write_text(json.dumps({'case':key,'returncode':rc,'reason':'guard protocol stop'}));break
 else:
  print('case not verified',key,rc,flush=True)
  if (p/'representative.json').exists():
   d=json.loads((p/'representative.json').read_text())
   if d.get('status')=='failed' and d.get('stage') not in ['runtime_import']:
    (T/'BATCH_STOP.json').write_text(json.dumps({'case':key,'returncode':rc,'reason':'canonical failure requires diagnosis'}));break
