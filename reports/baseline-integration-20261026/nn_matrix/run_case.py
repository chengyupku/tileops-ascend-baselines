from pathlib import Path
import os,sys,json,runpy,tempfile
P=Path(__file__).resolve().parent
W=Path('/root/dyq/workspace');R=W/'TileOPs'
m=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
assert m['phase']=='running' and m['child']['pid']==os.getpgrp()
op='GemmINT8W8A8FwdOp';cid='llama3-70b-prefill-down-bfloat16'
assert ['ops-nn',op,cid] in [x['key'] for x in json.loads((P/'SCOPE.json').read_text())]
out=P/'run';out.mkdir(exist_ok=False);os.chdir(out);short=Path(f'/proc/{os.getpid()}/cwd')
for n in ['tmp','ASCEND_CACHE_PATH','ASCEND_WORK_PATH','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR','ASCEND_PROCESS_LOG_PATH']:
 (out/n).mkdir();os.environ[n]=str(short/n)
os.environ['TMPDIR']=str(short/'tmp');tempfile.tempdir=str(short/'tmp')
v=str(W/'.cache/all-pairs/ops_nn/install3/vendors/allpairs_nn')
os.environ['OPS_NN_VENDOR']=v;os.environ['ASCEND_CLASSIC_NN_VENDOR']=v;os.environ['ASCEND_CUSTOM_OPP_PATH']=v
os.environ['ASCEND_GLOBAL_LOG_LEVEL']='0';os.environ['BASELINE_PROGRESS_FILE']=str(out/'progress.json')
sys.path[:0]=[str(P),str(R),str(R/'src'),str(W)];sys.dont_write_bytecode=True
from benchmarks.ascend_all_pairs import ops_nn
from adapter import int8_gemm
ops_nn.PROVIDERS[op]=int8_gemm
sys.argv=[str(R/'scripts/ascend_pair_check.py'),'--op',op,'--case-id',cid,'--library','ops-nn','--output',str(out/'checker.json')]
try:runpy.run_path(sys.argv[0],run_name='__main__')
finally:
 (out/'maps.txt').write_text(Path('/proc/self/maps').read_text())
 import torch
 if torch.npu.is_initialized():torch.npu.synchronize()
