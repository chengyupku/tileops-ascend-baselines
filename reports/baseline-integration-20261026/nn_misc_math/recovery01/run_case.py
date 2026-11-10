from pathlib import Path
import os,sys,json,runpy,hashlib,tempfile
R=Path(__file__).resolve().parent; W=Path('/root/dyq/workspace'); T=W/'TileOPs'
baseline,op,cid,output=sys.argv[1:]
assert [baseline,op,cid] in [r['key'] for r in json.loads((R.parent/'SCOPE.json').read_text())]
assert op!='SoftmaxFwdOp', 'Preserve completed historical case'
m=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert m.get('phase')=='running' and m.get('child',{}).get('pid')==os.getpgrp()
work=Path(output).resolve();work.mkdir(parents=True,exist_ok=False);os.chdir(work);short=Path(f'/proc/{os.getpid()}/cwd')
for name in ['tmp','ASCEND_CACHE_PATH','ASCEND_WORK_PATH','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR','ASCEND_PROCESS_LOG_PATH']:
 (work/name).mkdir();os.environ[name]=str(short/name)
os.environ['TMPDIR']=str(short/'tmp');tempfile.tempdir=str(short/'tmp');sys.dont_write_bytecode=True
os.environ['BASELINE_PROGRESS_FILE']=str(work/'progress.json');os.environ['ASCEND_GLOBAL_LOG_LEVEL']='0'
if baseline=='ops-nn':
 vendor=W/'.cache/all-pairs/ops_nn/install3/vendors/allpairs_nn';os.environ['OPS_NN_VENDOR']=str(vendor)
else:
 vendor=W/'.cache/all-pairs/ops_math/graph-runtime/vendors/allpairs_math_math';os.environ.pop('OPS_NN_VENDOR',None)
os.environ['ASCEND_CUSTOM_OPP_PATH']=str(vendor)
sys.path[:0]=[str(T),str(T/'src'),str(W),str(R)]
import benchmarks.ascend_all_pairs as packages
packages.__path__.insert(0,str(R/'providers'))
import importlib
provider=importlib.import_module('benchmarks.ascend_all_pairs.'+baseline.replace('-','_'))
assert Path(provider.__file__).resolve().is_relative_to(R/'providers')
if baseline=='ops-nn':
 from benchmarks.ascend_all_pairs.ops_nn import graph,normalization
 graph.WORKSPACE=short
 factory=normalization.count_nonzero if op=='CountNonzeroFwdOp' else normalization.normalization
else:
 from benchmarks.ascend_all_pairs.ops_math import graph
 from sum_adapter import make as factory
provider.PROVIDERS[op]=factory
(work/'ROUTE.json').write_text(json.dumps({'key':[baseline,op,cid],'provider':provider.__file__,'factory':str(factory),'vendor':str(vendor),'checker':str(T/'scripts/ascend_pair_check.py')},indent=2))
pin=json.loads((R/'CHECKER_PIN.json').read_text());assert hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest()==pin['sha256']
from native_log_capture import NativeLogCapture
os.environ['OPS_MATH_GRAPH_CALLS']=str(work/'graph_calls.jsonl')
sys.argv=[pin['path'],'--op',op,'--case-id',cid,'--library',baseline,'--output',str(work/'checker.json')]
try:
 with NativeLogCapture(work/'ASCEND_PROCESS_LOG_PATH',work/'native.jsonl'):runpy.run_path(pin['path'],run_name='__main__')
finally:
 import torch
 if torch.npu.is_initialized():torch.npu.synchronize()
 maps=Path('/proc/self/maps').read_text();(work/'maps.txt').write_text(maps)
