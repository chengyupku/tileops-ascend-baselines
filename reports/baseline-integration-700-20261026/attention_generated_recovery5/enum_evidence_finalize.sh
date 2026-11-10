python3 - <<'PY'
from pathlib import Path
import json,hashlib
r=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5');p=r/'HEADER_FAILURE_EVIDENCE.json';d=json.loads(p.read_text());sdk=Path('/usr/local/Ascend/cann-9.1.0/include/acl/acl_base_rt.h');s=sdk.read_text();d['failure']='Original flash_attn wrapper common/aclnn_common.h uses ACL_HIFLOAT4, absent from SDK aclDataType definition included by acl_base.h';d['enum_definition']={'path':str(sdk),'sha256':hashlib.sha256(sdk.read_bytes()).hexdigest(),'ACL_HIFLOAT4_present':'ACL_HIFLOAT4' in s,'ACL_HIFLOAT8_present':'ACL_HIFLOAT8' in s,'excerpt':'\n'.join(s.splitlines()[150:166])};p.write_text(json.dumps(d,indent=2));print(d['enum_definition'])
PY
