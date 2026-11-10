"""Optional future parent-run validation recipe; worker never executes this file."""
import sys,os,json,importlib.util
from pathlib import Path
ROOT=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_native9_registration')
W=Path('/opt/tileops-all-pairs/workspace')
sys.path[:0]=[str(ROOT),str(W/'TileOPs'),str(W/'TileOPs/src'),str(W)]
import registration as M
op,cid=sys.argv[1:]
assert (op,cid) in {(r['op'],r['case_id']) for r in M.rows()}
M.admit()
# All imports that may initialize device are below guarded environment/pin gates.
from benchmarks.ascend_all_pairs.ops_transformer import PROVIDERS
M.install(PROVIDERS)
p=W/'TileOPs/scripts/ascend_pair_check.py'
spec=importlib.util.spec_from_file_location('native9_canonical_checker',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
run=Path(os.environ['NATIVE9_RUN_ROOT'])
sys.argv=[str(p),'--op',op,'--case-id',cid,'--library','ops-transformer','--output',str(run/'representative.json')]
raise SystemExit(m.main())
