S=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt03/stage/source
python3 - <<'PY'
from pathlib import Path
b=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt03/stage/source/build')
for p in (b/'CMakeFiles').glob('*/build.make'):
 cmds=[l.strip() for l in p.read_text().splitlines() if l.startswith('\t') and not any(s in l for s in ['cmake_clean','cmake_depends','cmake_echo_color'])]
 if any('FlashAttn' in l or 'ascendc_ops_config' in l for l in cmds):print(p.parent.name,cmds)
PY
grep -nE 'ccache|exists.*\.o|os.path.exists|makefile' /usr/local/Ascend/cann-9.1.0/python/site-packages/asc_op_compile_base/asc_op_compiler/ascendc_compile_v220.py | head -40
head -35 "$S/cmake/scripts/util/const_var.py"
grep -nE 'execute_process|file\(INSTALL|include\(' "$S/build/cmake_install.cmake" | head -30
