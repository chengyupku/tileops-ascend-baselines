python3 - <<'PY'
from pathlib import Path
import json,hashlib
r=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5');s=r.parent/'attention_native_build5/attempt03/stage/source'
e=json.loads((r/'RELEVANT_EXPECTED_SOURCE_PINS.json').read_text());a={k:hashlib.sha256((s/k).read_bytes()).hexdigest() for k in e};bad=[k for k in e if e[k]!=a[k]]
(r/'SOURCE_PINS_RECHECK.json').write_text(json.dumps({'checked':len(a),'mismatches':bad,'actual':a,'basis':'parent frozen original source pins; only relevant compile/API/CMake files rehashed'},indent=2));print('checked',len(a),'mismatches',bad);assert not bad
PY
