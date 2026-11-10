import sys,runpy,json,datetime,os
from pathlib import Path
P=Path(__file__).resolve().parent
# Match the installed Torch 2.13 headers/libraries used by the extension build.
if os.environ.get('INTEGRATION700_GELU_BASE_RUNTIME') != '1':
    env=os.environ.copy();env['INTEGRATION700_GELU_BASE_RUNTIME']='1'
    env['LD_LIBRARY_PATH']='/opt/tileops-ci/runtime/lib/python3.12/site-packages/torch/lib:/opt/tileops-ci/runtime/lib/python3.12/site-packages/torch_npu/lib:'+env.get('LD_LIBRARY_PATH','')
    os.execve('/opt/tileops-ci/runtime/bin/python',['python','-I','-S','-B',str(Path(__file__).resolve())],env)
sys.path[:0]=['/opt/tileops-ci/runtime/lib/python3.12/site-packages','/usr/local/Ascend/cann-9.1.0/python/site-packages','/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe']
(P/'guard_snapshot.json').write_text(Path('/run/lock/ascend-npu/device-0.lock').read_text())
W=Path('/opt/tileops-all-pairs/workspace')
sys.path[:0]=[str(W/'TileOPs'),str(W/'TileOPs/src'),str(W),str(P)]
from benchmarks.ascend_all_pairs.ops_transformer import PROVIDERS
from integration700_transformer_dense_gelu import gelu
PROVIDERS['GeluTanhAndMulFwdOp']=gelu
sys.argv=[str(W/'TileOPs/scripts/ascend_pair_check.py'),'--op','GeluTanhAndMulFwdOp','--case-id','llama-8b-ffn-prefill-float16','--library','ops-transformer','--output',str(P/'checker')]
runpy.run_path(sys.argv[0],run_name='__main__')
