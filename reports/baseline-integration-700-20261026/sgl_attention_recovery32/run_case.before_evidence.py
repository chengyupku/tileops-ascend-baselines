from pathlib import Path
import os,sys,json,runpy,hashlib,tempfile
R=Path(__file__).resolve().parent; W=Path('/opt/tileops-all-pairs/workspace'); T=W/'TileOPs'
baseline,op,cid,output=sys.argv[1:]
scope=json.loads((R/'SCOPE.json').read_text()); assert [baseline,op,cid] in [[r['baseline'],r['operator'],r['case_id']] for r in scope]
m=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text()); assert m.get('phase')=='running' and m.get('child',{}).get('pid')==os.getpgrp()
work=Path(output).resolve(); work.mkdir(parents=True,exist_ok=False); os.chdir(work); short=Path(f'/proc/{os.getpid()}/cwd')
(work/'GUARD_RUNNING.json').write_text(json.dumps(m,indent=2))
for name in ['tmp','ASCEND_CACHE_PATH','ASCEND_WORK_PATH','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR','ASCEND_PROCESS_LOG_PATH']:
 (work/name).mkdir(); os.environ[name]=str(short/name)
os.environ['ASCEND_PROCESS_LOG_PATH']=os.environ.setdefault('SGL_CAPTURE_ROOT',str(work/'ASCEND_PROCESS_LOG_PATH'))
os.environ['TMPDIR']=str(short/'tmp'); tempfile.tempdir=str(short/'tmp'); sys.dont_write_bytecode=True
os.environ['BASELINE_PROGRESS_FILE']=str(work/'progress.json'); os.environ['ASCEND_GLOBAL_LOG_LEVEL']='0'
os.environ['ASCEND_LAUNCH_BLOCKING']='1'
sys.path[:0]=[str(T),str(T/'src'),str(W),str(R)]
sys.path.extend(['/opt/tileops-ci/runtime/lib/python3.12/site-packages',str(W/'.cache/all-pairs/remote_native_tla/python-deps'),str(W/'.cache/all-pairs/full_rollout_20261024/remote_ready/server2/group1/recovery_1622/tasks/fla-bf16/runtime/deps')])
import importlib.util
if 'workloads.numerics' in sys.modules:
 private=sys.modules['workloads.numerics']; assert Path(private.__file__).resolve()==R/'numerics_private.py'
else:
 spec=importlib.util.spec_from_file_location('workloads.numerics',R/'numerics_private.py')
 private=importlib.util.module_from_spec(spec); sys.modules['workloads.numerics']=private; spec.loader.exec_module(private)
override=json.loads((R/'CPU_DIAGNOSTIC_PREFLIGHT.json').read_text())
assert hashlib.sha256((T/'workloads/numerics.py').read_bytes()).hexdigest()==override['original_sha256']
assert hashlib.sha256((R/'numerics_private.py').read_bytes()).hexdigest()==override['private_sha256']
override['loaded_path']=private.__file__; override['loaded_compare_outputs_path']=private.compare_outputs.__code__.co_filename
override['kind']='diagnostic-only override: max_abs_err CPU; validators unchanged'
(work/'DIAGNOSTIC_OVERRIDE.json').write_text(json.dumps(override,indent=2))
import benchmarks.ascend_baselines as base
base.LIBRARIES=R/'libraries'
import benchmarks.ascend_all_pairs as packages
import importlib
module_name={'MLAKVCacheWriteFwdOp':'integration700_sgl_attention32','PagedKVCacheGatherFwdOp':'gather_adapter','DSADecodeWithKVCacheFwdOp':'dsa_adapter','GQAVarlenFwdOp':'attention_page16','GQAPrefillPagedWithKVCacheFwdOp':'prefill_adapter'}.get(op,'attention_adapter')
provider=importlib.import_module(module_name)
factory=provider.PROVIDERS[op]
from route_install import install
install(provider)
(work/'ROUTE.json').write_text(json.dumps({'key':[baseline,op,cid],'provider':provider.__file__,'factory':str(factory),'checker':str(T/'scripts/ascend_pair_check.py')},indent=2))
pin=json.loads((R/'CHECKER_PIN.json').read_text()); assert hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest()==pin['sha256']
from native_log_capture import NativeLogCapture
os.environ['OPS_MATH_GRAPH_CALLS']=str(work/'graph_calls.jsonl')
sys.argv=[pin['path'],'--op',op,'--case-id',cid,'--library',baseline,'--output',str(work/'checker.json')]
try:
 with NativeLogCapture(Path(os.environ.get('SGL_CAPTURE_ROOT',str(work/'ASCEND_PROCESS_LOG_PATH'))),work/'native.jsonl'): runpy.run_path(pin['path'],run_name='__main__')
finally:
 receipt=work/'checker.json'
 if receipt.exists():
  row=json.loads(receipt.read_text()); row['diagnostic_only_override']=override
  row.setdefault('source_sha256',{})['TileOPs/workloads/numerics.py']=override['original_sha256']
  row.setdefault('source_sha256',{})[str(R/'numerics_private.py')]=override['private_sha256']
  row['numerics_runtime_path']=private.__file__
  receipt.write_text(json.dumps(row,indent=2))
 import torch
 if torch.npu.is_initialized(): torch.npu.synchronize()
 (work/'maps.txt').write_text(Path('/proc/self/maps').read_text())
