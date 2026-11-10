python3 - <<'PY'
from pathlib import Path
import re,json
b=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt03/stage/source/build')
s=(b/'cmake_install.cmake').read_text()
for line in s.splitlines():
 if 'file(INSTALL' in line and 'OPTIONAL' not in line and ' FILES ' in line:
  paths=re.findall('"([^"]+)"',line.split(' FILES ',1)[1]);print([(p,Path(p).exists()) for p in paths])
print('metadata definitions', 'FlashAttnMetadata' in (b/'autogen/aic-ascend950-ops-info.json').read_text())
PY
