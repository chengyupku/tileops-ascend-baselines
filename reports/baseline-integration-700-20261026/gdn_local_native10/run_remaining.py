import json,subprocess,sys
from pathlib import Path
W=Path(__file__).resolve().parent
sys.path.insert(0,str(W));from audit_case import audit
cases=json.loads((W/'EXECUTION_ORDER.json').read_text())
for c in cases:
 d=W/c
 if (d/'GUARD_RECEIPT.json').exists():
  audit(c)
  continue
 p=subprocess.run([sys.executable,str(W/'run_stage.py'),c,'300',str(d/'child.sh')])
 if p.returncode:
  print('STOP_FOR_ANALYSIS',c,p.returncode,flush=True);sys.exit(p.returncode)
 result=audit(c)
 print('VERIFIED',c,result['verification'],flush=True)
