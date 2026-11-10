set -e
R=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5
npu-guard status --json > "$R/FINAL_GUARD_STATUS.json"
python3 - <<'PY'
from pathlib import Path
import json,hashlib,zipfile
r=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5');o=r.parent/'attention_native_build5';source=o/'attempt03/stage/source'
h=r/'python/cann_ops_transformer/common/aclnn_common.h';wheel=o/'attempt02/stage/source/build_out/cann_ops_transformer-1.0.0-py3-none-any.whl'
with zipfile.ZipFile(wheel) as z:wh=z.read('cann_ops_transformer/common/aclnn_common.h')
assert h.read_bytes()==wh
candidates=list((source/'torch_extension').rglob('aclnn_common.h'));print('original headers',candidates)
for p in candidates:assert p.read_bytes()==wh
sdk=Path('/usr/local/Ascend/cann-9.1.0/include/acl/acl_base.h');print('SDK header',sdk,sdk.exists())
text=sdk.read_text();print('ACL_HIFLOAT4 present', 'ACL_HIFLOAT4' in text,'ACL_HIFLOAT8 present','ACL_HIFLOAT8' in text)
row={'failure':'original flash_attn C++ wrapper compilation: ACL_HIFLOAT4 absent in installed SDK acl_base.h','installed_header_sha256':hashlib.sha256(wh).hexdigest(),'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),'matches_original_wheel':True,'matches_original_source_headers':[str(p) for p in candidates],'sdk_header':str(sdk),'sdk_header_sha256':hashlib.sha256(sdk.read_bytes()).hexdigest(),'ACL_HIFLOAT4_in_sdk_header':'ACL_HIFLOAT4' in text,'ACL_HIFLOAT8_in_sdk_header':'ACL_HIFLOAT8' in text,'native_flashattn_launched':False,'remaining_four_executed':False,'source_or_sdk_patch':False}
(r/'HEADER_FAILURE_EVIDENCE.json').write_text(json.dumps(row,indent=2))
n=json.loads((r/'gpt-oss-sink-decode-bfloat16-bfloat16.native.json').read_text());print('native launch records',n['launch']);print('loaded custom opapi paths',[p for p in n['loaded_files_sha256'] if 'cust_opapi' in p]);print('guard state',json.loads((r/'FINAL_GUARD_STATUS.json').read_text())['state'])
PY
