import sys,runpy
from pathlib import Path
root=Path('/root/dyq/workspace')
sys.path[:0]=[str(root/'TileOPs'),str(root/'TileOPs/src'),str(root)]
from benchmarks.ascend_all_pairs import fla
from integration700_fla_native import build
op,cid,out=sys.argv[1:]
fla.PROVIDERS[op]=build
sys.argv=['ascend_pair_check','--op',op,'--case-id',cid,'--library','flash-linear-attention-npu','--output',out]
runpy.run_path(str(root/'.cache/all-pairs/full_rollout_20261024/local24/group2/recovery_1155/fla_gdn_cpu_repair/ascend_pair_check_fla_gdn_cpu.py'),run_name='__main__')
