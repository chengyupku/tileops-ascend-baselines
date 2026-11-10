from pathlib import Path
import json,subprocess
w=Path(__file__).resolve().parent;b=json.loads((w/'BLOCKERS.json').read_text());groups=json.loads((w/'BATCH_PLAN.json').read_text())['groups'];removed=[]
for group in groups:
 for i in group:
  k='GDNFwdOp/'+i
  if k in b:removed.append({'key':k,'previous':b.pop(k),'fresh_resource_evidence':i+'/fresh_resource_admission.json'})
(w/'RESOURCE_READMISSION.json').write_text(json.dumps(removed,indent=2));(w/'BLOCKERS.json').write_text(json.dumps(b,indent=2));print('readmitted',len(removed),'batch cases',sum(map(len,groups)))
subprocess.run(['python3',str(w/'update_result.py')],check=True)
