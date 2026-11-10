S=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt03/stage/source
python3 - <<'PY'
from pathlib import Path
b=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt03/stage/source/build')
for n in ['optiling_compat','gen_version_info','modify_vendor','prepare_build']:
 p=b/f'CMakeFiles/{n}.dir/build.make'
 print(n)
 print('\n'.join(l for l in p.read_text().splitlines() if l.startswith('\t') and not any(s in l for s in ['cmake_clean','cmake_depends','cmake_echo_color'])))
PY
ls "$S/attention/flash_attn_metadata"
grep -n 'FlashAttnMetadata\|flash_attn_metadata' "$S/attention/flash_attn/op_host/op_api/"* | head -15
