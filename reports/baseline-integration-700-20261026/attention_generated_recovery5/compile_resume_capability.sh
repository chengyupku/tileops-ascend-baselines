python3 - <<'PY'
from pathlib import Path
root=Path('/usr/local/Ascend/cann-9.1.0/python/site-packages')
for p in root.glob('**/ascendc_compile_base.py'):print(p)
for p in root.glob('**/ascendc_compile.py'):print(p)
PY
ps -eo pid,etime,comm | grep -E 'bisheng|opc|make' | tail -10
