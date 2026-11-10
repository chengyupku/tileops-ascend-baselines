python3 - <<'PY'
from pathlib import Path
import json,datetime
r=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5')
for p in (r/'checkpoint_make_receipts').glob('*.json'):
 d=json.loads(p.read_text());print(d['makefile'].split('/')[-1],d['reuse_count'],d['reason'])
g=r.parent/'attention_native_build5/attempt03/stage/source/build/binary/ascend950/gen'
s={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':'checkpoint_resume','aiv':len(list(g.glob('kernel_meta_*/kernel_meta/*mix_aiv*.o'))),'aic':len(list(g.glob('kernel_meta_*/kernel_meta/*mix_aic*.o')))};(r/'HEARTBEAT.json').write_text(json.dumps(s));print(s)
PY
tail -4 /root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5/compile1_checkpoint.log
