from pathlib import Path
import subprocess,sys,json
P=Path(__file__).resolve().parent
scope=json.loads((P/'SCOPE.json').read_text())
for row in scope:
 if row['operator']!='GQADenseFwdOp' or row['case_id']=='qwen3-30b-1k-float16':continue
 cid=row['case_id']
 active={tuple(json.loads(f.read_text())['key']) for f in (P/'attempts').glob('*/command.json') if not (f.parent/'guard.json').exists()}
 if tuple(row['key']) in active:continue
 for host in ['local','openlibing-DevEnv-661495','openlibing-DevEnv-055409']:
  guards=[json.loads(f.read_text()) for f in (P/'attempts').glob('*/guard.json') if json.loads(f.read_text())['key']==row['key']]
  if any(g['returncode'] not in [75,76,78] for g in guards):break
  if host in [g['host'] for g in guards]:continue
  subprocess.run([sys.executable,str(P/'orchestrate.py'),cid,host],check=True)
  subprocess.run([sys.executable,str(P/'summarize.py')],check=True)
