python3 - <<'PY'
from pathlib import Path
import hashlib,json
r=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5');v=r/'native_install/vendors/attention_native_build5_transformer'
rows=[]
for p in v.rglob('*'):
 if p.is_file() and (p.suffix in ['.so','.o','.json'] or p.name=='version.info'):
  rows.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size':p.stat().st_size})
(r/'NATIVE_ARTIFACTS.json').write_text(json.dumps(rows,indent=2));print('artifacts',len(rows))
PY
tail -8 /root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5/gpt-oss-sink-decode-bfloat16-bfloat16.log
