import sys,os,json,runpy,tempfile
from pathlib import Path
P=Path(__file__).resolve().parent;W=Path('/opt/tileops-all-pairs/workspace');out=P/'topk_check_graph';out.mkdir(exist_ok=True)
(out/'guard_snapshot.json').write_text(Path('/run/lock/ascend-npu/device-0.lock').read_text())
vendor=P/'selection/selectionpairs_transformer';os.environ['ASCEND_CUSTOM_OPP_PATH']=str(vendor);os.environ['ASCEND_CLASSIC_TRANSFORMER_VENDOR']=str(vendor);os.environ['TRANSFORMER_ATTENTION_VENDOR']=str(vendor)
for name in ['ASCEND_CACHE_PATH','ASCEND_WORK_PATH','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR','ASCEND_PROCESS_LOG_PATH','TRITON_CACHE_DIR']:
 d=out/name;d.mkdir(exist_ok=True);os.environ[name]=str(d)
os.environ['ASCEND_GLOBAL_LOG_LEVEL']='0'
sys.path[:0]=[str(W/'TileOPs'),str(W/'TileOPs/src'),str(W),str(P)]
sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
from benchmarks.ascend_all_pairs.ops_transformer import PROVIDERS
from integration700_transformer_dense_topk import topk
for cid in ['qwen3-235b-t32-bfloat16','qwen3-235b-t512-bfloat16']:
 op='FusedTopKFwdOp'
 PROVIDERS[op]=topk
 sys.argv=[str(W/'TileOPs/scripts/ascend_pair_check.py'),'--op',op,'--case-id',cid,'--library','ops-transformer','--output',str(out/(cid+'.json'))]
 try:runpy.run_path(sys.argv[0],run_name='__main__')
 except SystemExit as e:
  if e.code:raise
