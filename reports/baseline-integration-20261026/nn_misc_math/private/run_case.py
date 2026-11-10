import os,sys,json,hashlib,runpy,tempfile
from pathlib import Path
# args baseline op case output
baseline,op,cid,out=sys.argv[1:]
W=Path('/root/dyq/workspace'); P=Path(__file__).resolve().parent; T=P/'TileOPs'; work=Path(out).resolve(); work.mkdir(parents=True,exist_ok=False)
for n in ['tmp','ASCEND_CACHE_PATH','ASCEND_WORK_PATH','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR','ASCEND_PROCESS_LOG_PATH']:
 (work/n).mkdir(); os.environ[n]=str(work/n)
os.environ['TMPDIR']=str(work/'tmp'); os.environ['PYTHONDONTWRITEBYTECODE']='1'; os.environ['ASCEND_GLOBAL_LOG_LEVEL']='0'; os.environ['BASELINE_PROGRESS_FILE']=str(work/'progress.json')
os.environ['OPS_NN_VENDOR']='/root/dyq/workspace/.tools/baselines/opp/ops-nn/vendors/baseline_nn_nn'
os.environ['ASCEND_CUSTOM_OPP_PATH']=os.environ['OPS_NN_VENDOR']
sys.path[:0]=[str(T),str(T/'src'),str(W),str(P)]
import torch
import benchmarks.ascend_all_pairs.registry as reg
# Ensure exact private provider package is used and all five cases route through original staging adapter.
mod=__import__('benchmarks.ascend_all_pairs.'+reg.LIBRARY_MODULES[baseline],fromlist=['x'])
# Provider public gating is enabled by accepted_cases/enabled_cases edits above.
# Invoke copied checker in this same guarded process.
sys.argv=[str(T/'ascend_pair_check.py'),'--op',op,'--case-id',cid,'--library',baseline,'--output',str(work/'checker.json')]
runpy.run_path(sys.argv[0],run_name='__main__')
