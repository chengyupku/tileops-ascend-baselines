"""Private original-API binding; complete canonical checker, under host guard only."""
from pathlib import Path
import os,sys,json,runpy,tempfile,hashlib,importlib.util
P=Path(__file__).resolve().parent
W=Path('/root/dyq/workspace');V=W/'.cache/all-pairs/full_rollout_20261024/repairs/resume_20261025/boundary_1557/tla_pristine_build/source'
C=V.parent
R=P/'runtime'
op,cid=sys.argv[1:3];key=['tilelang-ascend',op,cid]
assert key in [[r['baseline'],r['operator'],r['case_id']] for r in json.loads((P/'SCOPE.json').read_text())]
meta=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
assert meta['phase']=='running' and meta['child']['pid']==os.getpgrp() and meta['owner']=='baseline700-20261026/threehost_pto_original23'
out=P/'runs'/(op+'_'+cid+(sys.argv[3] if len(sys.argv)>3 else ''));out.mkdir(parents=True,exist_ok=False);os.chdir(out)
(out/'GUARD_RUNNING.json').write_text(json.dumps(meta,indent=2))
for n in ['ASCEND_NPU_ARCH','TVM_HOME','TVM_IMPORT_PYTHON_PATH']:os.environ.pop(n,None)
os.environ.update(TL_PLATFORM='A5',TL_ROOT=str(V),TL_TEMPLATE_PATH=str(V/'src'),TVM_IMPORT_PYTHON_PATH=str(V/'3rdparty/tvm/python'),TILELANG_LIBRARY_PATH=str(V/'build'),TVM_LIBRARY_PATH=str(V/'build/tvm'),SETUPTOOLS_USE_DISTUTILS='local',PYTHONDONTWRITEBYTECODE='1')
for n in ['TILELANG_CACHE_DIR','TMPDIR','XDG_CACHE_HOME','TLA_CYTHON_CACHE_DIR','ASCEND_WORK_PATH','ASCEND_CACHE_PATH','ASCEND_PROCESS_LOG_PATH','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR']:
 d=P/'cache'/n;d.mkdir(parents=True,exist_ok=True);os.environ[n]=str(d)
tempfile.tempdir=os.environ['TMPDIR'];sys.dont_write_bytecode=True
os.environ['PATH']=str(W/'.venv/bin')+':'+os.environ['PATH']
sys.path[:0]=[str(R),str(V/'3rdparty/tvm/python'),str(C/'venv/lib/python3.12/site-packages'),str(W/'.venv/lib/python3.12/site-packages'),str(P),str(P/'frozen'),str(P/'frozen/src'),str(W/'TileOPs/src'),str(W)]

sdk=Path('/usr/local/Ascend/cann-9.2.0-beta.2')
os.environ.update(ASCEND_HOME_PATH=str(sdk),ASCEND_TOOLKIT_HOME=str(sdk),BISHENG_HOME=str(sdk/'tools/bisheng_compiler'))
os.environ['PATH']=str(sdk/'tools/bisheng_compiler/bin')+':'+os.environ['PATH']
import setuptools
from private_tla import native,deployment
# Existing deployment hook changes only cache location; no compiler/kernel change.
# Private unchanged package owns its original .cycache location.
import tilelang,torch
assert Path(tilelang.__file__).resolve()==(R/'tilelang/__init__.py').resolve()
native.VENDOR=P/'original'
native.runtime=lambda:tilelang
from benchmarks.ascend_all_pairs import tla
from benchmarks.ascend_baselines import Candidate
from benchmarks.api import Implementation
from adapter import factory

def bind(case):
 return factory(case,op,native,tilelang,out)
tla.PROVIDERS[op]=bind
from observe import install
install(out,tilelang,torch)
checker=P/'frozen/scripts/ascend_pair_check.py'
sys.argv=[str(checker),'--op',op,'--case-id',cid,'--library','tilelang-ascend','--output',str(out/'checker.json')]
try:runpy.run_path(str(checker),run_name='__main__')
finally:
 if torch.npu.is_initialized():torch.npu.synchronize()
 (out/'maps.txt').write_text(Path('/proc/self/maps').read_text())
