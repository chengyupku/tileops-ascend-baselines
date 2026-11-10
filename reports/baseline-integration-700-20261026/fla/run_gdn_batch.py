import sys,runpy,json,gc
from pathlib import Path
root=Path('/root/dyq/workspace');p=Path(__file__).resolve().parent
sys.path[:0]=[str(root/'TileOPs'),str(root/'TileOPs/src'),str(root)]
from benchmarks.ascend_all_pairs import fla
from integration700_fla_native import build
fla.PROVIDERS['GDNFwdOp']=build
checker=runpy.run_path(str(root/'.cache/all-pairs/full_rollout_20261024/local24/group2/recovery_1155/fla_gdn_cpu_repair/ascend_pair_check_fla_gdn_cpu.py'))
ids=sys.argv[1:]
for cid in ids:
 sys.argv=['ascend_pair_check','--op','GDNFwdOp','--case-id',cid,'--library','flash-linear-attention-npu','--output',str(p/f'GDNFwdOp.{cid}.checker.json')]
 rc=checker['main']();gc.collect()
 if rc:raise SystemExit(rc)
