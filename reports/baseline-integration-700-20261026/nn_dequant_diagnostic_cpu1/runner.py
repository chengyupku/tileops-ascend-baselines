import os,sys,json,runpy,tempfile
from pathlib import Path
P=Path(__file__).resolve().parent;W=Path('/root/dyq/workspace');T=W/'TileOPs'
base,op,cid,attempt=sys.argv[1:];assert any((r['baseline'],r['operator'],r['case_id'])==(base,op,cid) for r in json.loads((P/'SCOPE.json').read_text()))
m=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert m['phase']=='running' and m['child']['pid']==os.getpgrp() and m['owner']=='baseline700-20261026/nn_dequant_diagnostic_cpu1'
out=P/attempt;out.mkdir(exist_ok=True);(out/'GUARD_METADATA.json').write_text(json.dumps(m,indent=2));os.chdir(out);short=Path(f'/proc/{os.getpid()}/cwd');sys.dont_write_bytecode=True
for n in ['tmp','ASCEND_CACHE_PATH','ASCEND_WORK_PATH','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR','ASCEND_PROCESS_LOG_PATH']:
 (out/n).mkdir(exist_ok=True);os.environ[n]=str(short/n)
os.environ['TMPDIR']=str(short/'tmp');tempfile.tempdir=str(short/'tmp');os.environ['BASELINE_PROGRESS_FILE']=str(out/'progress.json');os.environ['ASCEND_GLOBAL_LOG_LEVEL']='0'
os.environ['NN_MATH_BASELINE']=base
if base=='ops-nn':vendor=W/'.cache/all-pairs/ops_nn/install3/vendors/allpairs_nn';os.environ['ASCEND_CLASSIC_NN_VENDOR']=str(vendor);os.environ['OPS_NN_VENDOR']=str(vendor)
else:vendor=W/'.cache/ascend-classic/native-pilot-20261020/ops-math/install/vendors/classicpilot_math';os.environ['ASCEND_CLASSIC_MATH_VENDOR']=str(vendor)
if op in ('GemmFwdOp','GroupedGemmFwdOp'):
 vendor=W/'.cache/ascend-classic/native-pilot-20261020/ops-nn/install/vendors/classicpilot_nn';os.environ['OPS_NN_VENDOR']=str(vendor)
if op in ('INT8DequantPerChannelFwdOp',):vendor=W/'.cache/all-pairs/ops_math/graph-runtime/vendors/allpairs_math_math'
os.environ['ASCEND_CUSTOM_OPP_PATH']=str(vendor)
sys.path[:0]=[str(T),str(T/'src'),str(W),str(P),str(W/'reports/baseline-integration-20261026/nn_misc_math/recovery01')]
import importlib,importlib.util,hashlib,time
started=time.time()
spec=importlib.util.spec_from_file_location('workloads.numerics',P/'numerics_private.py')
private=importlib.util.module_from_spec(spec);sys.modules['workloads.numerics']=private;spec.loader.exec_module(private)
override=json.loads((P/'CPU_DIAGNOSTIC_PREFLIGHT.json').read_text())
assert hashlib.sha256((T/'workloads/numerics.py').read_bytes()).hexdigest()==override['original_sha256']
assert hashlib.sha256((P/'numerics_private.py').read_bytes()).hexdigest()==override['private_sha256']
override.update(loaded_path=private.__file__,loaded_compare_outputs_path=private.compare_outputs.__code__.co_filename)
(out/'DIAGNOSTIC_OVERRIDE.json').write_text(json.dumps(override,indent=2))
os.environ['OPS_MATH_GRAPH_CALLS']=str(out/'graph_calls.jsonl')
provider=importlib.import_module('benchmarks.ascend_all_pairs.'+base.replace('-','_'))
sys.path.insert(0,str(out))
from adapter import make
provider.PROVIDERS[op]=make
from native_log_capture import NativeLogCapture
sys.argv=[str(T/'scripts/ascend_pair_check.py'),'--op',op,'--case-id',cid,'--library',base,'--output',str(out/'checker.json')]
try:
 with NativeLogCapture(out/'ASCEND_PROCESS_LOG_PATH',out/'native.jsonl'):runpy.run_path(sys.argv[0],run_name='__main__')
finally:
 receipt=out/'checker.json'
 if receipt.exists():
  row=json.loads(receipt.read_text());row['diagnostic_only_override']=override;row['numerics_runtime_path']=private.__file__
  row.setdefault('source_sha256',{})[str(P/'numerics_private.py')]=override['private_sha256']
  receipt.write_text(json.dumps(row,indent=2))
 (out/'CHILD_TIMING.json').write_text(json.dumps({'started_epoch':started,'finished_epoch':time.time(),'pid':os.getpid(),'pgid':os.getpgrp()},indent=2))
 import torch
 (out/'LOADED_MAPS.txt').write_text(Path('/proc/self/maps').read_text())
 (out/'LOADED_MODULES.json').write_text(json.dumps({n:getattr(m,'__file__',None) for n,m in sys.modules.items()},indent=2))

 if torch.npu.is_initialized():torch.npu.synchronize()
