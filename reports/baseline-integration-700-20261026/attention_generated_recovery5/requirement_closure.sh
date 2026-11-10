export PYTHONDONTWRITEBYTECODE=1
R=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5
O=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5
PYTHONPATH="$R/python:$O/attempt01/stage/python:/opt/tileops-ci/runtime/lib/python3.12/site-packages" /opt/tileops-ci/runtime/bin/python3 - <<'PY'
import pathlib,importlib.metadata as m,json
from packaging.requirements import Requirement
r=pathlib.Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5');source=r.parent/'attention_native_build5/attempt03/stage/source'
rows=[]
for text in (source/'requirements.txt').read_text().splitlines()+['ml-dtypes==0.5.4','numpy<2.0,>=1.26.0']:
 if not text.strip() or text.startswith('#'):continue
 q=Requirement(text)
 try:
  d=m.distribution(q.name);rows.append({'requirement':text,'version':d.version,'satisfied':d.version in q.specifier,'path':str(d._path)})
 except m.PackageNotFoundError:rows.append({'requirement':text,'satisfied':False})
(r/'DEPENDENCY_CPU_REVIEW.json').write_text(json.dumps({'checks':rows,'all_satisfied':all(x['satisfied'] for x in rows),'runtime_abi':'unproven; no native imports performed','private_change':'ml-dtypes 0.6.0 numpy>=2 conflict repaired by private 0.5.4; original numpy1.26.4 reused'},indent=2))
print(json.dumps(rows,indent=2))
PY
