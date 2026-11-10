"""Private original-API binding; complete canonical checker, under host guard only."""
from pathlib import Path
import os,sys,json,runpy,tempfile,hashlib,importlib.util
P=Path(__file__).resolve().parent
W=Path('/opt/tileops-all-pairs/workspace');V=P/'original'
C=W/'.cache/all-pairs/remote_tla_compiler'
R=W/'.cache/all-pairs/full_rollout_20261024/remote_ready/server1/group1/recovery_resume25/tla_supplemental_1858/runtime_env2/runtime/package'
op,cid=sys.argv[1:];key=['tilelang-ascend',op,cid]
assert key in [[r['baseline'],r['operator'],r['case_id']] for r in json.loads((P/'SCOPE.json').read_text())]
meta=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
assert meta['phase']=='running' and meta['child']['pid']==os.getpgrp() and meta['owner']=='baseline700-20261026/surge_tla_attention97/worker05'
out=P/'runs'/(cid+'-pto-v3');out.mkdir(parents=True,exist_ok=True);os.chdir(out)
(out/'GUARD_RUNNING.json').write_text(json.dumps(meta,indent=2))
for n in ['ASCEND_NPU_ARCH','TVM_HOME','TVM_IMPORT_PYTHON_PATH']:os.environ.pop(n,None)
os.environ.update(TL_PLATFORM='A5',TL_ROOT=str(W/"ascend_baselines/tilelang-ascend"),TILELANG_LIBRARY_PATH=str(C/'handoff/lib'),TVM_LIBRARY_PATH=str(C/'handoff/lib'),SETUPTOOLS_USE_DISTUTILS='local',PYTHONDONTWRITEBYTECODE='1')
for n in ['TILELANG_CACHE_DIR','TMPDIR','XDG_CACHE_HOME','TLA_CYTHON_CACHE_DIR','ASCEND_WORK_PATH','ASCEND_CACHE_PATH','ASCEND_PROCESS_LOG_PATH','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR']:
 d=(P/'shared_cache'/n) if n in ['TLA_CYTHON_CACHE_DIR','TILELANG_CACHE_DIR'] else out/n;d.mkdir(parents=True,exist_ok=True);os.environ[n]=str(d)
tempfile.tempdir=os.environ['TMPDIR'];sys.dont_write_bytecode=True
os.environ['PATH']=str(C/'handoff/bin')+':'+os.environ['PATH']
sys.path[:0]=[str(R),str(W/'ascend_baselines/tilelang-ascend/3rdparty/tvm/python'),str(C/'python'),str(W/'.cache/all-pairs/remote_native_tla/python-deps'),str(P),str(W/'TileOPs'),str(W/'TileOPs/src'),str(W)]
sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
import setuptools
from private_tla import native,deployment
# Existing deployment hook changes only cache location; no compiler/kernel change.
deployment.install_cache_hook(R,Path(os.environ['TLA_CYTHON_CACHE_DIR']))
(out/'IDENTITY_BEFORE_IMPORT.json').write_text((P/'CPU_IDENTITY.json').read_text())
import tilelang,torch
assert Path(tilelang.__file__).resolve()==(R/'tilelang/__init__.py').resolve()
native.VENDOR=V
native.runtime=lambda:tilelang
from benchmarks.ascend_all_pairs import tla
from benchmarks.ascend_baselines import Candidate
from benchmarks.api import Implementation
from adapter import factory

def bind(case):
 return factory(case,op,native,tilelang,out)
tla.PROVIDERS[op]=bind
checker=W/'TileOPs/scripts/ascend_pair_check.py'
sys.argv=[str(checker),'--op',op,'--case-id',cid,'--library','tilelang-ascend','--output',str(out/'checker.json')]
try:runpy.run_path(str(checker),run_name='__main__')
finally:
 if torch.npu.is_initialized():torch.npu.synchronize()
 (out/'maps.txt').write_text(Path('/proc/self/maps').read_text())
