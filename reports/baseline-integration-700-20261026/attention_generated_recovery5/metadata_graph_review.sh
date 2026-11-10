S=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt03/stage/source
python3 - <<'PY'
from pathlib import Path
b=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt03/stage/source/build')
p=b/'attention/flash_attn_metadata/op_kernel_aicpu/CMakeFiles/flash_attn_metadata_cust_obj.dir/build.make'
print('metadata object command')
print('\n'.join(l for l in p.read_text().splitlines() if l.startswith('\t') and any(s in l for s in ['g++','cmake_progress_start'])))
for n in ['gen_aicpu_json_from_opdef','merge_aicpu_json']:
 p=b/f'CMakeFiles/{n}.dir/build.make';print(n);print('\n'.join(l for l in p.read_text().splitlines() if l.startswith('\t') and not any(s in l for s in ['cmake_echo','cmake_clean','cmake_depends'])))
PY
