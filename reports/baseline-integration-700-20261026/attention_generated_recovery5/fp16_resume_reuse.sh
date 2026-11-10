python3 - <<'PY'
from pathlib import Path
import json,datetime
r=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5')
for p in sorted((r/'checkpoint_make_receipts').glob('*.json'))[-2:]:
 d=json.loads(p.read_text());print(d['makefile'].split('/')[-1],d['reuse_count'])
g=r.parent/'attention_native_build5/attempt03/stage/source/build/binary/ascend950/gen';print(datetime.datetime.now(datetime.timezone.utc).isoformat(),len(list(g.glob('kernel_meta_*/kernel_meta/*mix_aic*.o'))))
PY
tail -3 /root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5/compile0_checkpoint.log
