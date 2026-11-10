"""Future execution child only: all imports and JIT inside host npu-guard."""
from pathlib import Path
import os, sys, json, hashlib, runpy, importlib.util, tempfile
P=Path(__file__).resolve().parent
op,cid,output=sys.argv[1:]
assert ['ops-nn',op,cid] in json.loads((P/'EXACT_KEYS.json').read_text())
m=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
assert m.get('phase')=='running' and m.get('child',{}).get('pid')==os.getpgrp(), 'Whole process must be inside npu-guard'
W=Path('/root/dyq/workspace'); R=W/'TileOPs'; work=Path(output).resolve()
assert work.parent.parent==P.parent
work.mkdir(parents=True,exist_ok=False)
os.chdir(work); short=Path(f'/proc/{os.getpid()}/cwd')
for name in ['tmp','ASCEND_CACHE_PATH','ASCEND_WORK_PATH','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR','ASCEND_PROCESS_LOG_PATH']:
 (work/name).mkdir();os.environ[name]=str(short/name)
os.environ['TMPDIR']=str(short/'tmp');tempfile.tempdir=str(short/'tmp')
os.environ['BASELINE_PROGRESS_FILE']=str(work/'progress.json')
os.environ['ASCEND_GLOBAL_LOG_LEVEL']='0';os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
vendor=json.loads((P/'VENDORS.json').read_text())[op]
os.environ['OPS_NN_VENDOR']=vendor;os.environ['ASCEND_CUSTOM_OPP_PATH']=vendor
sys.path[:0]=[str(R),str(R/'src'),str(W),str(P)]
from native_log_capture import NativeLogCapture
from benchmarks.ascend_all_pairs import ops_nn
from benchmarks.ascend_all_pairs.ops_nn import graph
graph.WORKSPACE=short  # Cache destination only; public compiler options unchanged.
def frozen(name, filename):
 spec=importlib.util.spec_from_file_location(name,P/'frozen'/filename)
 mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
if op=='RMSNormFwdOp':
 frozen('benchmarks.ascend_graph_call','ascend_graph_call.py')
 frozen('benchmarks.ascend_nn_direct','ascend_nn_direct.py')
 adapter=frozen('benchmarks._frozen_rms','ascend_rms_cases.py').opsnn_rms_cases
else:
 adapter=frozen('benchmarks.ascend_all_pairs.ops_nn._frozen_norm','normalization.py').normalization
if op=='InstanceNormFwdOp' and cid=='image-eval-float16':
 adapter=frozen('benchmarks.ascend_all_pairs.ops_nn._private_instance_eval','instance_eval.py').instance_eval
if op=='InstanceNormFwdOp' and cid=='wider-channel-affine-float16':
 adapter=frozen('benchmarks.ascend_all_pairs.ops_nn._private_instance_train','instance_train.py').instance_train
ops_nn.PROVIDERS[op]=adapter
sys.argv=[str(R/'scripts/ascend_pair_check.py'),'--op',op,'--case-id',cid,'--library','ops-nn','--output',str(work/'checker.json')]
try:
 with NativeLogCapture(work/'ASCEND_PROCESS_LOG_PATH',work/'native.jsonl'):
  runpy.run_path(sys.argv[0],run_name='__main__')
finally:
 import torch
 if torch.npu.is_initialized():torch.npu.synchronize()
 maps=Path('/proc/self/maps').read_text();(work/'maps.txt').write_text(maps)
 artifacts={}
 for line in maps.splitlines():
  path=line.split()[-1]
  if path.startswith('/') and any(v in path for v in ('libcust_','libacl_op_compiler','libascendcl','liboptiling')) and Path(path).is_file():artifacts[path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
 (work/'loaded_artifacts.json').write_text(json.dumps(artifacts,indent=2)+'\n')
