import sys,os,json,runpy
from pathlib import Path
P=Path(__file__).resolve().parent;W=Path('/opt/tileops-all-pairs/workspace');out=P/'quant_siblings';out.mkdir(exist_ok=True)
(out/'guard_snapshot.json').write_text(Path('/run/lock/ascend-npu/device-0.lock').read_text())
vendor=P/'gemm/private_provider';os.environ['ASCEND_CUSTOM_OPP_PATH']=str(vendor);os.environ['ASCEND_CLASSIC_TRANSFORMER_VENDOR']=str(vendor)
for name in ['ASCEND_CACHE_PATH','ASCEND_WORK_PATH','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR','ASCEND_PROCESS_LOG_PATH','TRITON_CACHE_DIR']:
 d=out/name;d.mkdir(exist_ok=True);os.environ[name]=str(d)
os.environ['ASCEND_GLOBAL_LOG_LEVEL']='0'
sys.path[:0]=[str(W/'TileOPs'),str(W/'TileOPs/src'),str(W),str(P)]
sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
from benchmarks.ascend_all_pairs.ops_transformer import PROVIDERS
from integration700_transformer_dense_quant import quant
import time
started=time.monotonic()
keys=[x['case_id'] for x in json.loads((P/'SCOPE.json').read_text()) if x['operator']=='GemmINT8W8A8FwdOp' and x['case_id']!='llama3-70b-dec-qkv-tsr-bfloat16']
keys.sort(key=lambda cid:('prefill' in cid,cid))
for cid in keys:
 if time.monotonic()-started>210:break
 op='GemmINT8W8A8FwdOp';PROVIDERS[op]=quant
 sys.argv=[str(W/'TileOPs/scripts/ascend_pair_check.py'),'--op',op,'--case-id',cid,'--library','ops-transformer','--output',str(out/(cid+'.json'))]
 try:runpy.run_path(sys.argv[0],run_name='__main__')
 except SystemExit as e:
  if e.code:break
