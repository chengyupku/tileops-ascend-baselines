R=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5
python3 - <<'PY'
from pathlib import Path
import json
r=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5');p=r/'gpt-oss-sink-decode-bfloat16-bfloat16.checker.json'
if p.exists():
 d=json.loads(p.read_text());print(json.dumps({k:d.get(k) for k in ['status','stage','error_type','error','elapsed_seconds','verification']},indent=2))
s=(r/'gpt-oss-sink-decode-bfloat16-bfloat16.log').read_text();lines=s.splitlines()
for i,l in enumerate(lines):
 if 'ACL_HIFLOAT4' in l or ('flash_attn.cpp' in l and 'error:' in l):print(i,l[:600])
print('last lines');print('\n'.join(lines[-8:]))
PY
