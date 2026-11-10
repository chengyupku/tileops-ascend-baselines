import sys, json, os, runpy
from pathlib import Path
P=Path(__file__).resolve().parent
guard=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
assert guard['phase']=='running' and guard['child']['pid']==os.getpgrp()
sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
cid=sys.argv[1]
out=P/('registered-check-'+cid+'-deps')
out.mkdir(exist_ok=False)
(out/'GUARD_RUNNING.json').write_text(json.dumps(guard,indent=2))
from benchmarks.ascend_all_pairs import tilelang as provider
from bmm_registration import install
install(provider.PROVIDERS)
checker=Path('/opt/tileops-all-pairs/workspace/TileOPs/scripts/ascend_pair_check.py')
sys.argv=[str(checker),'--op','BmmFwdOp','--case-id',cid,'--library','tilelang','--output',str(out/'checker.json')]
runpy.run_path(str(checker),run_name='__main__')
