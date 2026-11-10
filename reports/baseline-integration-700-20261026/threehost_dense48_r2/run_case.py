import os,sys,runpy,json
from pathlib import Path
P=Path(__file__).resolve().parent
for k in ['ASCEND_CACHE_PATH','ASCEND_WORK_PATH','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR','ASCEND_PROCESS_LOG_PATH','TRITON_CACHE_DIR']:
 d=P/'run'/k;d.mkdir(parents=True,exist_ok=True);os.environ[k]=str(d)
os.environ['ASCEND_CUSTOM_OPP_PATH']=str(P/'vendor');os.environ['ASCEND_CLASSIC_TRANSFORMER_VENDOR']=str(P/'vendor');os.environ['ASCEND_GLOBAL_LOG_LEVEL']='0'
sys.path[:0]=[str(P/'TileOPs'),str(P/'TileOPs/src'),str(P),'/opt/tileops-ci/runtime/lib/python3.12/site-packages']
(P/'guard_snapshot.json').write_text(Path('/run/lock/ascend-npu/device-0.lock').read_text())
from benchmarks.ascend_all_pairs.ops_transformer import PROVIDERS
from binding import build
PROVIDERS['GemmFP8FwdOp']=build
sys.argv=[str(P/'TileOPs/scripts/ascend_pair_check.py'),'--op','GemmFP8FwdOp','--case-id','ds-v3-m1-down-blk-float8_e4m3fn-bfloat16','--library','ops-transformer','--output',str(P/'checker.json')]
runpy.run_path(sys.argv[0],run_name='__main__')
