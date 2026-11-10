"""Private original-API binding; complete canonical checker, under host guard only."""
from pathlib import Path
import os,sys,json,runpy,tempfile,hashlib,importlib.util
P=Path(__file__).resolve().parent
W=Path('/root/dyq/workspace');V=W/'ascend_baselines/tilelang-ascend'
C=Path('/root/dyq/workspace/.cache/all-pairs/full_rollout_20261024/repairs/resume_20261025/boundary_1557/tla_pristine_build')
R=V
F=P/'frozen/TileOPs'
op,cid=sys.argv[1:];assert (op,cid)==('GQADenseFwdOp','qwen3-30b-4k-float16');key=['tilelang-ascend',op,cid]
assert key in [[r['baseline'],r['operator'],r['case_id']] for r in json.loads((P/'SCOPE.json').read_text())]
meta=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
assert meta['phase']=='running' and meta['child']['pid']==os.getpgrp() and meta['owner']=='baseline700-20261026/surge_tla_attention97/worker04'
out=P/'continuation_20261027T0440'/'local_runs'/(cid+'-pto-A5-original-direct');out.mkdir(parents=True,exist_ok=True);os.chdir(out)
(out/'GUARD_RUNNING.json').write_text(json.dumps(meta,indent=2))
for n in ['ASCEND_NPU_ARCH','TVM_HOME','TVM_IMPORT_PYTHON_PATH']:os.environ.pop(n,None)
os.environ['TL_PLATFORM']='A5'
os.environ['INTEGRATION700_TLA_ROUTE']='direct'
os.environ.update(TL_ROOT=str(V),TILELANG_LIBRARY_PATH=str(C/'source/build'),TVM_LIBRARY_PATH=str(C/'source/build/tvm'),SETUPTOOLS_USE_DISTUTILS='local',PYTHONDONTWRITEBYTECODE='1')
for n in ['TILELANG_CACHE_DIR','TMPDIR','XDG_CACHE_HOME','TLA_CYTHON_CACHE_DIR','ASCEND_WORK_PATH','ASCEND_CACHE_PATH','ASCEND_PROCESS_LOG_PATH','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR']:
 d=(P/'shared_cache'/n) if n in ['TLA_CYTHON_CACHE_DIR','TILELANG_CACHE_DIR'] else out/n;d.mkdir(parents=True,exist_ok=True);os.environ[n]=str(d)
tempfile.tempdir=os.environ['TMPDIR'];sys.dont_write_bytecode=True
os.environ['PATH']=str(Path(os.environ['ASCEND_HOME_PATH'])/'tools/bisheng_compiler/bin')+':'+os.environ['PATH']
sys.path[:0]=[str(R),str(V/'3rdparty/tvm/python'),str(C/'venv/lib/python3.11/site-packages'),str(W/'.cache/all-pairs/remote_native_tla/python-deps'),str(P),str(F),str(F/'src'),str(W)]
sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
import setuptools
sys.path.insert(0,str(P))
from private_tla import native,deployment

# Existing deployment hook changes only cache location; no compiler/kernel change.
deployment.install_cache_hook(R,Path(os.environ['TLA_CYTHON_CACHE_DIR']))
import tilelang,torch
(out/'import_maps.txt').write_text(Path('/proc/self/maps').read_text())
(out/'runtime_identity.json').write_text(json.dumps({'tilelang':tilelang.__file__,'torch':torch.__file__,'sys_path':sys.path,'environment':{k:v for k,v in os.environ.items() if k.startswith(('ASCEND','TL_','TILELANG','TVM','LD_'))}},indent=2))
assert Path(tilelang.__file__).resolve()==(R/'tilelang/__init__.py').resolve()
native.VENDOR=V
native.runtime=lambda:tilelang
from benchmarks.ascend_all_pairs import tla
from benchmarks.ascend_baselines import Candidate
from benchmarks.api import Implementation
if op == 'AllFwdOp':
 from other_adapter import factory
else:
 from adapter_direct import factory
from capture import install, snapshot
install(out, tilelang, torch)
snapshot(out, "before_checker")
source_paths=[V/'testing/python/language/test_tilelang_ascend_language_elementwise.py',P/'other_adapter.py'] if op=='AllFwdOp' else [V/'examples/gqa_fwd_varlen/gqa_fwd_varlen.py',P/'adapter.py']
(out/'preuse_sources.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_paths},indent=2))

def bind(case):
 return factory(case,op,native,tilelang,out)
tla.PROVIDERS[op]=bind
checker=F/'scripts/ascend_pair_check.py'
sys.argv=[str(checker),'--op',op,'--case-id',cid,'--library','tilelang-ascend','--output',str(out/'checker.json')]
try:runpy.run_path(str(checker),run_name='__main__')
finally:
 (out/'maps.txt').write_text(Path('/proc/self/maps').read_text())
 if torch.npu.is_initialized():torch.npu.synchronize()
