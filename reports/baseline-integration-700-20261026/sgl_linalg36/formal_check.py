"""After parent registration only: original checker, no provider mutation."""
import pathlib,sys,runpy
P=pathlib.Path(__file__).resolve().parent;W=pathlib.Path('/opt/tileops-all-pairs/workspace')
sys.path.extend(['/opt/tileops-ci/runtime/lib/python3.12/site-packages',str(W/'.cache/all-pairs/remote_native_tla/python-deps'),str(W/'.cache/all-pairs/full_rollout_20261024/remote_ready/server2/group1/recovery_1622/tasks/fla-bf16/runtime/deps')])
checker=W/'TileOPs/scripts/ascend_pair_check.py';sys.argv=[str(checker)]+sys.argv[1:]
runpy.run_path(str(checker),run_name='__main__')
