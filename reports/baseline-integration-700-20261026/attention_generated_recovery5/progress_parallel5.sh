python3 - <<'PY'
from pathlib import Path
import datetime,json,hashlib
r=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5');g=r.parent/'attention_native_build5/attempt03/stage/source/build/binary/ascend950/gen'
s={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':'parallel_compile','objects':len(list(g.glob('kernel_meta_*/kernel_meta/*.o')))}
print(s);(r/'HEARTBEAT.json').write_text(json.dumps(s))
for p in g.glob('kernel_meta_*/kernel_meta/*.i'):print(p.name,hashlib.sha256(p.read_bytes()).hexdigest())
PY
tail -3 /root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5/compile1_parallel.log
