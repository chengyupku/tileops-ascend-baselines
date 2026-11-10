from pathlib import Path
import sys,os,json,hashlib,runpy,time,tempfile
P=Path(__file__).resolve().parent; W=Path('/opt/tileops-all-pairs/workspace'); T=W/'TileOPs'
lib,op,cid=sys.argv[1:]
assert lib=='ops-nn' and op=='FusedAddLayerNormFwdOp' and cid in ('llama-70b-prefill-float16','llama-8b-prefill-float16')
guard=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
assert guard['phase']=='running' and guard['child']['pid']==os.getpgrp() and guard['owner']=='single-server1/fp64'
out=P/'runs_addln_original'/lib/cid;out.mkdir(parents=True,exist_ok=False)
os.chdir(out)
(out/'GUARD_RUNNING.json').write_text(json.dumps(guard,indent=2))
for name in ['TMPDIR','ASCEND_CACHE_PATH','ASCEND_WORK_PATH','ASCEND_PROCESS_LOG_PATH','TRITON_CACHE_DIR','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR']:
 (out/name).mkdir();os.environ[name]=str(Path('/proc/self/cwd')/name)
tempfile.tempdir=os.environ['TMPDIR']
vendor=Path(json.loads((P/'NN_DEPLOYMENT.json').read_text())['vendor'])
os.environ['OPS_NN_VENDOR']=str(vendor)
os.environ['ASCEND_CUSTOM_OPP_PATH']=str(vendor)
sys.path[:0]=[str(P),str(T),str(T/'src'),str(W)]
sys.path.extend(['/opt/tileops-ci/runtime/lib/python3.12/site-packages',str(W/'.cache/all-pairs/remote_native_tla/python-deps'),str(W/'.cache/all-pairs/remote_native_vllm/python-deps')])
import torch
torch.set_num_threads(4);torch.set_num_interop_threads(4)
from benchmarks.ascend_all_pairs.registry import LIBRARY_MODULES
import importlib
registry=importlib.import_module('benchmarks.ascend_all_pairs.'+LIBRARY_MODULES[lib])
if cid=='llama-70b-prefill-float16':
 from nn_adapter_capture import build
else:
 from nn_adapter import build
original_build=build
source_before={}
def factory(case):
 c=original_build(case)
 source_before.update({str(Path(k).resolve()):v for k,v in c.source_sha256.items() if Path(k).is_file()})
 (out/'CANDIDATE_SOURCE_BEFORE.json').write_text(json.dumps(source_before,indent=2))
 return c
registry.PROVIDERS[op]=factory
oracle_paths=[T/'workloads/norm.py',T/'workloads/numerics.py',T/'scripts/ascend_pair_check.py']
shared={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in oracle_paths}
(out/'SHARED_BEFORE.json').write_text(json.dumps(shared,indent=2))
# Trace function entry/return only, without changing original call signatures or code.
events=open(out/'PHASES.jsonl','a',buffering=1)
allow={'gdn_ref','kda_ref','validate','chunk_gated_delta_rule_native','chunk_gated_delta_rule_npu','chunk_kda','npu_chunk_kda_fwd','reference','candidate'}
def trace(frame,event,arg):
 if event in ('call','return') and frame.f_code.co_name in allow:
  events.write(json.dumps({'time':time.time(),'event':event,'function':frame.f_code.co_name,'file':frame.f_code.co_filename})+'\n')
sys.setprofile(trace)
sys.argv=[str(T/'scripts/ascend_pair_check.py'),'--op',op,'--case-id',cid,'--library',lib,'--output',str(out/'checker.json')]
try:runpy.run_path(str(T/'scripts/ascend_pair_check.py'),run_name='__main__')
finally:
 sys.setprofile(None);events.close()
 if torch.npu.is_initialized():torch.npu.synchronize()
 changed={k:hashlib.sha256(Path(k).read_bytes()).hexdigest() for k,v in {**shared,**source_before}.items() if Path(k).is_file() and hashlib.sha256(Path(k).read_bytes()).hexdigest()!=v}
 (out/'SOURCE_AFTER.json').write_text(json.dumps({'changed':changed},indent=2))
 maps=Path('/proc/self/maps').read_text();(out/'maps.txt').write_text(maps)
 loaded={}
 for name,mod in tuple(sys.modules.items()):
  f=getattr(mod,'__file__',None)
  if f and Path(f).is_file() and name.startswith(('sgl_kernel_npu','fla_npu','_allpairs_vllm','vllm_private','workloads.')):
   loaded[name]={'path':f,'sha256':hashlib.sha256(Path(f).read_bytes()).hexdigest()}
 (out/'LOADED_MODULES.json').write_text(json.dumps(loaded,indent=2))
 binaries={}
 for line in maps.splitlines():
  f=line.split()[-1]
  if f.startswith('/') and '.so' in f and ('ops-nn' in f or 'single_s1_nn' in f or str(out) in f) and Path(f).is_file():binaries[f]=hashlib.sha256(Path(f).read_bytes()).hexdigest()
 (out/'BINARIES.json').write_text(json.dumps(binaries,indent=2))
 assert not changed,changed
