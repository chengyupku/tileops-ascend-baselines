R=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5
B=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt03/stage/source/build
python3 - <<'PY'
from pathlib import Path
p=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt03/stage/source/build/binary/ascend950/gen')
for m in p.glob('kernel_meta_*/kernel_meta/*mk'):
 t=m.read_text();print(m.name,'bytes',len(t),'object targets',sum(1 for l in t.splitlines() if '.o:' in l),'objects present',len(list(m.parent.glob('*.o'))))
PY
tail -3 "$R/compile1.log"
