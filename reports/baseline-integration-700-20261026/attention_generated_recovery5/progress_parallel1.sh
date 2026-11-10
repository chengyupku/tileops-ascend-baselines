python3 - <<'PY'
from pathlib import Path
import json,datetime
r=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5');b=r.parent/'attention_native_build5/attempt03/stage/source/build/binary/ascend950/gen'
s={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':'guarded_parallel_kernel_compile','objects':len(list(b.glob('kernel_meta_*/kernel_meta/*.o')))}
(r/'HEARTBEAT.json').write_text(json.dumps(s));print(s)
PY
ps -eo comm | sort | uniq -c | grep -E 'bisheng|make'
tail -3 /root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5/compile1_parallel.log
ls /usr/local/Ascend/cann-9.1.0/python/site-packages/asc_op_compile_base/asc_op_compiler | head -25
