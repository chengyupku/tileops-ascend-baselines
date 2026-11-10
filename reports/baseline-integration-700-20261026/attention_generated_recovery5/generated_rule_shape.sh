python3 - <<'PY'
from pathlib import Path
import shlex
p=next(Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt03/stage/source/build/binary/ascend950/gen').glob('kernel_meta_*/kernel_meta/*mk'))
t=p.read_text().splitlines();print('linecount',len(t))
for i,l in enumerate(t[:6]):print(i,l[:200], '...tail...',l[-200:])
for l in t:
 if l.startswith('\t'):
  a=shlex.split(l);print('recipe executable',a[0],'output',a[a.index('-o')+1]);break
PY
