from pathlib import Path
import os,sys,json,runpy,hashlib,tempfile
R=Path(__file__).resolve().parent; W=Path('/opt/tileops-all-pairs/workspace'); T=W/'TileOPs'
op,cid,label=sys.argv[1:]; baseline="sgl-kernel-npu"; output=str(R/"runs"/(label+"--"+op+"--"+cid)/"output")
scope=json.loads((R/'SCOPE.json').read_text()); assert [baseline,op,cid] in [[r['baseline'],r['operator'],r['case_id']] for r in scope]
m=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text()); assert m.get('phase')=='running' and m.get('child',{}).get('pid')==os.getpgrp()
work=Path(output).resolve(); work.mkdir(parents=True,exist_ok=False); os.chdir(work); short=Path(f'/proc/{os.getpid()}/cwd')
(work/'GUARD_RUNNING.json').write_text(json.dumps(m,indent=2))
for name in ['tmp','ASCEND_CACHE_PATH','ASCEND_WORK_PATH','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR','ASCEND_PROCESS_LOG_PATH']:
 (work/name).mkdir(); os.environ[name]=str(short/name)
os.environ['TMPDIR']=str(short/'tmp'); tempfile.tempdir=str(short/'tmp'); sys.dont_write_bytecode=True
os.environ['BASELINE_PROGRESS_FILE']=str(work/'progress.json'); os.environ['ASCEND_GLOBAL_LOG_LEVEL']='0'

sys.path[:0]=[str(T),str(T/'src'),str(W),str(R)]
sys.path.extend(['/opt/tileops-ci/runtime/lib/python3.12/site-packages',str(W/'.cache/all-pairs/remote_native_tla/python-deps'),str(W/'.cache/all-pairs/full_rollout_20261024/remote_ready/server2/group1/recovery_1622/tasks/fla-bf16/runtime/deps')])
sys.path.append(str(W/'.cache/all-pairs/remote_native_vllm/python-deps'))
import time
(work.parent/'acquired.json').write_text(json.dumps({'utc_epoch':time.time(),'metadata':m}))
os.environ.pop('ASCEND_CUSTOM_OPP_PATH',None)
(work/'ENVIRONMENT.json').write_text(json.dumps({'custom_opp':os.environ.get('ASCEND_CUSTOM_OPP_PATH'),'numerics':'shared original','reason':'parent FEEDBACK; no custom OPP required for SGL native'}))
import benchmarks.ascend_all_pairs as packages
import importlib
provider=importlib.import_module('integration700_sgl_gdn_l2' if cid=='prefill-l2norm-ragged-bfloat16' else 'integration700_sgl_gdn_remote38')
factory=provider.PROVIDERS[op]
from route_install import install
install(provider)
(work/'ROUTE.json').write_text(json.dumps({'key':[baseline,op,cid],'provider':provider.__file__,'factory':str(factory),'checker':str(T/'scripts/ascend_pair_check.py')},indent=2))
pin=json.loads((R/'CHECKER_PIN.json').read_text()); assert hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest()==pin['sha256']
from native_log_capture import NativeLogCapture
os.environ['OPS_MATH_GRAPH_CALLS']=str(work/'graph_calls.jsonl')
sys.argv=[pin['path'],'--op',op,'--case-id',cid,'--library',baseline,'--output',str(work/'checker.json')]
try:
 with NativeLogCapture(Path(os.environ.setdefault('GDN_NATIVE_LOG_ROOT',str(work/'ASCEND_PROCESS_LOG_PATH'))),work/'native.jsonl'): runpy.run_path(pin['path'],run_name='__main__')
finally:
 receipt=work/'checker.json'
 if receipt.exists(): (work.parent/'checker.json').write_bytes(receipt.read_bytes())
 (work/'LOADED_MODULES.json').write_text(json.dumps({name:{'path':getattr(mod,'__file__',None),'sha256':hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest() if getattr(mod,'__file__',None) and Path(mod.__file__).is_file() else None} for name,mod in list(sys.modules.items()) if name.startswith(('sgl_kernel_npu','workloads.numerics','fla'))},indent=2))
 import torch
 if torch.npu.is_initialized(): torch.npu.synchronize()
 (work/'maps.txt').write_text(Path('/proc/self/maps').read_text())
