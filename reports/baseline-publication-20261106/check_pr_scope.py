import json,subprocess,yaml
from pathlib import Path
P=Path('/root/dyq/workspace/reports/baseline-publication-20261106');R='/tmp/tileops-ci-baseline2208-20261106';C=Path('/root/dyq/workspace/reports/torch-all2208-20261105/source/src/tileops/manifest/spec')
scopes=json.loads((P/'pr-operator-scope.json').read_text());out=[]
for x in scopes:
 if x['pr']==42:x['ops']=['FusedTopKFwdOp','MoEGroupedGemmFP8FwdOp']
 specs={}
 for f in C.glob('*.yaml'):
  data=yaml.safe_load(f.read_text()) or {}
  if not set(x['ops'])&data.keys():continue
  raw=subprocess.check_output(['git','-C',R,'show',x['head']+':src/tileops/manifest/spec/'+f.name],text=True);prspec=yaml.safe_load(raw)
  for op in set(x['ops'])&data.keys():
   a=data[op];b=prspec[op];specs[op]={'workloads_equal':a['workloads']==b['workloads'],'signature_equal':a['signature']==b['signature'],'manifest':f.name}
 x['spec_checks']=specs;out.append(x)
 print(x['pr'],specs)
(P/'pr-operator-scope.json').write_text(json.dumps(out,indent=2))
