"""Private original-API binding; complete canonical checker, under host guard only."""
from pathlib import Path
import os,sys,json,runpy,tempfile,hashlib,importlib.util
P=Path(__file__).resolve().parent
W=Path('/opt/tileops-all-pairs/workspace');V=W/'ascend_baselines/tilelang-ascend'
C=W/'.cache/all-pairs/remote_tla_compiler'
R=V
F=P/'frozen/TileOPs'
op,cid=sys.argv[1:];key=['tilelang-ascend',op,cid]
assert key in [[r['baseline'],r['operator'],r['case_id']] for r in json.loads((P/'SCOPE.json').read_text())]
meta=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
assert meta['phase']=='running' and meta['child']['pid']==os.getpgrp() and meta['owner']=='baseline700-20261026/surge_tla_attention97/worker02'
out=P/'runs'/(cid+'-pto-A5-original-branch'+os.environ.get('ATTEMPT_SUFFIX',''));out.mkdir(parents=True,exist_ok=True);os.chdir(out)
(out/'GUARD_RUNNING.json').write_text(json.dumps(meta,indent=2))
for n in ['ASCEND_NPU_ARCH','TVM_HOME','TVM_IMPORT_PYTHON_PATH']:os.environ.pop(n,None)
os.environ['TL_PLATFORM']='A5'
os.environ.update(TL_ROOT=str(V),TILELANG_LIBRARY_PATH=str(C/'handoff/lib'),TVM_LIBRARY_PATH=str(C/'handoff/lib'),SETUPTOOLS_USE_DISTUTILS='local',PYTHONDONTWRITEBYTECODE='1')
for n in ['TILELANG_CACHE_DIR','TMPDIR','XDG_CACHE_HOME','TLA_CYTHON_CACHE_DIR','ASCEND_WORK_PATH','ASCEND_CACHE_PATH','ASCEND_PROCESS_LOG_PATH','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR']:
 d=(P/'shared_cache'/n) if n in ['TLA_CYTHON_CACHE_DIR','TILELANG_CACHE_DIR'] else out/n;d.mkdir(parents=True,exist_ok=True);os.environ[n]=str(d)
tempfile.tempdir=os.environ['TMPDIR'];sys.dont_write_bytecode=True
os.environ['PATH']=str(C/'handoff/bin')+':'+os.environ['PATH']
sys.path[:0]=[str(R),str(V/'3rdparty/tvm/python'),str(C/'python'),str(W/'.cache/all-pairs/remote_native_tla/python-deps'),str(P),str(F),str(F/'src'),str(W)]
sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
identities={}
for root in [V/'examples/flash_attention/flash_attn_bhsd.py', V/'examples/sparse_flash_attention/example_sparse_flash_attn_dynamic_shape.py', V/'examples/attention_sink/example_gqa_sink_fwd_varlen/example_gqa_sink_fwd_varlen.py', F/'scripts/ascend_pair_check.py', P/'adapter.py']:
 identities[str(root)]=hashlib.sha256(root.read_bytes()).hexdigest()
(out/'pre_use_identity.json').write_text(json.dumps(identities,indent=2))
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
from adapter import factory
from capture import install, snapshot
install(out, tilelang, torch)
snapshot(out, "before_checker")

def bind(case):
 seed_module=Path(sys.modules['tools.baseline_protocol'].__file__)
 (out/'checker_seed.json').write_text(json.dumps(dict(seed=torch.initial_seed(),module=str(seed_module),sha256=hashlib.sha256(seed_module.read_bytes()).hexdigest()),indent=2))
 return factory(case,op,native,tilelang,out)
tla.PROVIDERS[op]=bind
checker=F/'scripts/ascend_pair_check.py'
sys.argv=[str(checker),'--op',op,'--case-id',cid,'--library','tilelang-ascend','--output',str(out/'checker.json')]
try:runpy.run_path(str(checker),run_name='__main__')
finally:
 (out/'maps.txt').write_text(Path('/proc/self/maps').read_text())
 if torch.npu.is_initialized():torch.npu.synchronize()
