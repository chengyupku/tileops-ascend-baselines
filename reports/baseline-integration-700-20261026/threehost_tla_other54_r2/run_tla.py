from pathlib import Path
import os,sys,json,runpy,tempfile,time,hashlib,ctypes
P=Path(__file__).resolve().parent;W=Path('/opt/tileops-all-pairs/workspace');V=W/'ascend_baselines/tilelang-ascend';C=W/'.cache/all-pairs/remote_tla_compiler';R=W/'.cache/all-pairs/full_rollout_20261024/remote_ready/server1/group1/recovery_resume25/tla_supplemental_1858/runtime_env2/runtime/package'
baseline,op,cid=sys.argv[1:4];assert baseline=='tilelang-ascend' and op=='CausalConv1dFwdOp';assert [baseline,op,cid] in [[r['baseline'],r['operator'],r['case_id']] for r in json.loads((P/'SCOPE.json').read_text())]
meta=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert meta['owner']=='baseline700-20261026/threehost_tla_other54_r2' and meta['phase']=='running' and meta['child']['pid']==os.getpgrp()
out=P/'runs'/('_'.join([baseline,op,cid])+(sys.argv[4] if len(sys.argv)>4 else ''));out.mkdir(parents=True,exist_ok=False);os.chdir(out);(out/'GUARD_RUNNING.json').write_text(json.dumps(meta,indent=2))
for n in ['ASCEND_NPU_ARCH','TVM_HOME','TVM_IMPORT_PYTHON_PATH']:os.environ.pop(n,None)
os.environ.update(TL_PLATFORM='A5',TL_ROOT=str(V),TILELANG_LIBRARY_PATH=str(C/'handoff/lib'),TVM_LIBRARY_PATH=str(C/'handoff/lib'),SETUPTOOLS_USE_DISTUTILS='local',PYTHONDONTWRITEBYTECODE='1')
for n in ['TILELANG_CACHE_DIR','TMPDIR','XDG_CACHE_HOME','TLA_CYTHON_CACHE_DIR','ASCEND_WORK_PATH','ASCEND_CACHE_PATH','ASCEND_PROCESS_LOG_PATH','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR']:
 d=P/'tla_cache'/n;d.mkdir(parents=True,exist_ok=True);os.environ[n]=str(d)
tempfile.tempdir=os.environ['TMPDIR'];sys.dont_write_bytecode=True
sdk=W/'.cache/all-pairs/remote_native_deepgemm_deepep/sdk-920b2'
os.environ.update(ASCEND_HOME_PATH=str(sdk),ASCEND_TOOLKIT_HOME=str(sdk),BISHENG_HOME=str(sdk/'tools/bisheng_compiler'))
os.environ['PATH']=str(sdk/'tools/bisheng_compiler/bin')+':'+str(C/'handoff/bin')+':'+os.environ['PATH']
sys.path[:0]=[str(R),str(V/'3rdparty/tvm/python'),str(C/'python'),str(W/'.cache/all-pairs/remote_native_tla/python-deps'),str(P),str(P/'frozen'),str(W/'TileOPs/src'),str(W)]
import setuptools
from private_tla import native,deployment
deployment.install_cache_hook(R,Path(os.environ['TLA_CYTHON_CACHE_DIR']))
import tilelang,torch
native.VENDOR=V;native.runtime=lambda:tilelang
from benchmarks.ascend_all_pairs import tla
assert str(P/'frozen') in tla.__file__
from state_adapter_pto import factory
from tilelang.jit.kernel import JITKernel
old=JITKernel.__call__;seq=0
def trace(k,*args,**kwargs):
 global seq
 seq+=1;f=Path(k.adapter.lib._name);st=f.stat();data=f.read_bytes();sha=hashlib.sha256(data).hexdigest();dest=out/'native_artifacts'/sha;dest.parent.mkdir(exist_ok=True);dest.write_bytes(data)
 r=dict(path=str(f),binary_sha256=sha,device=st.st_dev,inode=st.st_ino,symbols={},maps=Path('/proc/self/maps').read_text(),target=str(k.target))
 for name in ['call','init']:
  try:r['symbols'][name]=ctypes.cast(getattr(k.adapter.lib,name),ctypes.c_void_p).value
  except Exception:pass
 (out/(str(seq)+'.dispatch.json')).write_text(json.dumps(r,indent=2));(out/(str(seq)+'.kernel.txt')).write_text(k.get_kernel_source());result=old(k,*args,**kwargs);torch.npu.synchronize();r['completed']=time.time();(out/(str(seq)+'.dispatch.json')).write_text(json.dumps(r,indent=2));return result
JITKernel.__call__=trace
tla.PROVIDERS[op]=lambda case:factory(case,op,native,tilelang,out)
checker=P/'frozen/scripts/ascend_pair_check.py';sys.argv=[str(checker),'--op',op,'--case-id',cid,'--library',baseline,'--output',str(out/'checker.json')]
try:runpy.run_path(str(checker),run_name='__main__')
finally:
 if torch.npu.is_initialized():torch.npu.synchronize()
