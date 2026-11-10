python3 - <<'PY'
from pathlib import Path
p=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt03/stage/source/build/binary/ascend950/gen')
for m in p.glob('kernel_meta_*/kernel_meta/*mk'):
 t=m.read_text().splitlines();print('targets',len(t[0].split())-1,'objects',len(list(m.parent.glob('*.o')))); print('rule sample',t[1][:500],t[2][:500])
PY
