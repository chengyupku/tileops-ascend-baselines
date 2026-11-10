python3 - <<'PY'
import json,pathlib
m=json.loads(pathlib.Path('/run/lock/ascend-npu/device-0.lock').read_text());print(m);print(pathlib.Path('/proc/%s/cmdline'%m['pid']).read_bytes())
PY
