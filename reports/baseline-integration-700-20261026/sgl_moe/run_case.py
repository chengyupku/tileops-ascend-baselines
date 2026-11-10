import os,sys,json,runpy,time
from pathlib import Path
P=Path(__file__).resolve().parent; W=Path('/root/dyq/workspace')
op,cid,attempt=sys.argv[1:]; a=Path(attempt).resolve(); a.mkdir(exist_ok=True)
sys.dont_write_bytecode=True
os.environ['MOE_ATTEMPT']=str(a)
os.environ['BASELINE_PROGRESS_FILE']=str(a/'progress.json')
for name in ['ASCEND_CACHE_PATH','ASCEND_WORK_PATH','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR','ASCEND_PROCESS_LOG_PATH','TMPDIR']:
    (a/name).mkdir(exist_ok=True);os.environ[name]=str(a/name)
os.chdir(a)
sys.path[:0]=[str(W/'TileOPs'),str(W/'TileOPs/src'),str(W),str(P)]
from benchmarks.ascend_all_pairs import sgl
import integration700_sgl_moe
sgl.PROVIDERS[op]=integration700_sgl_moe.build
sys.argv=[str(W/'TileOPs/scripts/ascend_pair_check.py'),'--op',op,'--case-id',cid,'--library','sgl-kernel-npu','--output',str(a/'checker.json')]
(a/'device_start.json').write_text(json.dumps({'time':time.time()}))
try: runpy.run_path(sys.argv[0],run_name='__main__')
finally:
    import torch,torch.distributed as dist
    if torch.npu.is_initialized():torch.npu.synchronize()
    if dist.is_initialized():dist.destroy_process_group()
