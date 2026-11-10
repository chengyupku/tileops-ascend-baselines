from pathlib import Path
import os,sys,json,runpy,hashlib,tempfile
R=Path(__file__).resolve().parent; W=Path('/root/dyq/workspace'); T=W/'TileOPs'
baseline,op,cid,output=sys.argv[1:]
scope=json.loads((R/'SCOPE.json').read_text()); assert [baseline,op,cid] in [[r['baseline'],r['operator'],r['case_id']] for r in scope]
m=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text()); assert m.get('phase')=='running' and m.get('child',{}).get('pid')==os.getpgrp()
work=Path(output).resolve(); work.mkdir(parents=True,exist_ok=False); os.chdir(work); short=Path(f'/proc/{os.getpid()}/cwd')
(work/'GUARD_RUNNING.json').write_text(json.dumps(m,indent=2))
for name in ['tmp','ASCEND_CACHE_PATH','ASCEND_WORK_PATH','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR','ASCEND_PROCESS_LOG_PATH']:
 (work/name).mkdir(); os.environ[name]=str(short/name)
os.environ['TMPDIR']=str(short/'tmp'); tempfile.tempdir=str(short/'tmp'); sys.dont_write_bytecode=True
os.environ['BASELINE_PROGRESS_FILE']=str(work/'progress.json'); os.environ['ASCEND_GLOBAL_LOG_LEVEL']='0'
os.environ['ASCEND_CUSTOM_OPP_PATH']=str(W/'.cache/all-pairs/ops_math/graph-runtime/vendors/allpairs_math_math')
sys.path[:0]=[str(T),str(T/'src'),str(W),str(R)]
import benchmarks.ascend_all_pairs as packages
import importlib
provider=importlib.import_module('integration700_sgl_gdn_recovery02')
factory=provider.PROVIDERS[op]
from route_install import install
install(provider)
(work/'ROUTE.json').write_text(json.dumps({'key':[baseline,op,cid],'provider':provider.__file__,'factory':str(factory),'checker':str(T/'scripts/ascend_pair_check.py')},indent=2))
pin=json.loads((R/'CHECKER_PIN.json').read_text()); assert hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest()==pin['sha256']
from native_log_capture import NativeLogCapture
os.environ['OPS_MATH_GRAPH_CALLS']=str(work/'graph_calls.jsonl')
sys.argv=[pin['path'],'--op',op,'--case-id',cid,'--library',baseline,'--output',str(work/'checker.json')]
try:
 with NativeLogCapture(work/'ASCEND_PROCESS_LOG_PATH',work/'native.jsonl'): runpy.run_path(pin['path'],run_name='__main__')
finally:
 import torch
 if torch.npu.is_initialized(): torch.npu.synchronize()
 (work/'maps.txt').write_text(Path('/proc/self/maps').read_text())
