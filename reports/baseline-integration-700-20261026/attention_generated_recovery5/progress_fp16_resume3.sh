python3 - <<'PY'
from pathlib import Path
import datetime
r=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5');g=r.parent/'attention_native_build5/attempt03/stage/source/build/binary/ascend950/gen';print(datetime.datetime.now(datetime.timezone.utc).isoformat(),'cube',len(list(g.glob('kernel_meta_*/kernel_meta/*mix_aic*.o'))))
PY
tail -2 /root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5/compile0_checkpoint.log
