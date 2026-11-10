from pathlib import Path
import os,sys,json,runpy,tempfile,time,hashlib,ctypes
P=Path(__file__).resolve().parent;W=Path('/opt/tileops-all-pairs/workspace');V=W/'ascend_baselines/tilelang-ascend';C=W/'.cache/all-pairs/remote_tla_compiler';R=P/'runtime'
baseline,op,cid=sys.argv[1:4];assert baseline=='tilelang-ascend' and op=='IsnanFwdOp';assert [baseline,op,cid] in [[r['baseline'],r['operator'],r['case_id']] for r in json.loads((P/'SCOPE.json').read_text())]
meta=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert meta['owner']=='baseline700-20261026/threehost_isnan_adapter1' and meta['phase']=='running' and meta['child']['pid']==os.getpgrp()
out=P/'runs'/('_'.join([baseline,op,cid])+(sys.argv[4] if len(sys.argv)>4 else ''));out.mkdir(parents=True,exist_ok=False);os.chdir(out);(out/'GUARD_RUNNING.json').write_text(json.dumps(meta,indent=2))
for n in ['ASCEND_NPU_ARCH','TVM_HOME','TVM_IMPORT_PYTHON_PATH']:os.environ.pop(n,None)
os.environ.update(TL_PLATFORM='A5',TL_ROOT=str(V),TL_TEMPLATE_PATH=str(V/'src'),TILELANG_LIBRARY_PATH=str(C/'handoff/lib'),TVM_LIBRARY_PATH=str(C/'handoff/lib'),SETUPTOOLS_USE_DISTUTILS='local',PYTHONDONTWRITEBYTECODE='1')
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
import subprocess
original_subprocess_run=subprocess.run
def capture_run(cmd,*args,**kwargs):
 if isinstance(cmd,(list,tuple)) and any('bisheng' in str(x) for x in cmd[:1]):
  (out/'compiler_command.json').write_text(json.dumps(dict(argv=[str(x) for x in cmd],env={n:os.environ.get(n) for n in ['TL_PLATFORM','TL_ROOT','TL_TEMPLATE_PATH','ASCEND_HOME_PATH','BISHENG_HOME']}),indent=2))
 return original_subprocess_run(cmd,*args,**kwargs)
subprocess.run=capture_run
import tilelang,torch
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
source=V/'testing/python/language/test_tilelang_ascend_language_elementwise.py'
assert digest(source)=='303b90f72b7b73b800f252e46cebf2dca682bea6b9362dd051f53d80044afb84'
evidence=dict(platform=os.environ['TL_PLATFORM'],runtime=tilelang.__file__,torch=torch.__version__,source_sha256=digest(source),sdk=str(sdk),files={})
for f in [sdk/'tools/bisheng_compiler/bin/bisheng',C/'handoff/bin/ptoas',*[C/'handoff/lib'/n for n in ['libtilelang.so','libtilelang_module.so','libtvm.so','libtvm_runtime.so']],R/'tilelang/jit/kernel.py',R/'tilelang/jit/adapter/cython/adapter.py']:
 if f.is_file():evidence['files'][str(f)]=dict(sha256=digest(f),inode=f.stat().st_ino)
(out/'deployment.json').write_text(json.dumps(evidence,indent=2))

native.VENDOR=V;native.runtime=lambda:tilelang;native.CACHE=P/"tla_cache"
from benchmarks.ascend_all_pairs import tla
assert str(P/'frozen') in tla.__file__
from element_adapter import factory
from tilelang.jit.kernel import JITKernel
old=JITKernel.__call__;seq=0
def trace(k,*args,**kwargs):
 global seq
 seq+=1;f=Path(k.adapter.lib._name);st=f.stat();data=f.read_bytes();sha=hashlib.sha256(data).hexdigest();dest=out/'native_artifacts'/sha;dest.parent.mkdir(exist_ok=True);dest.write_bytes(data)
 r=dict(path=str(f),binary_sha256=sha,device=st.st_dev,inode=st.st_ino,symbols={},maps=Path('/proc/self/maps').read_text(),target=str(k.target),phase='before_native_call',inputs=[dict(shape=list(x.shape),dtype=str(x.dtype),numel=x.numel(),ptr=x.data_ptr()) for x in args])
 for name in ['call','init']:
  try:r['symbols'][name]=ctypes.cast(getattr(k.adapter.lib,name),ctypes.c_void_p).value
  except Exception:pass
 kernel_source=k.get_kernel_source();(out/(str(seq)+'.kernel.txt')).write_text(kernel_source);r['generated_sha256']=hashlib.sha256(kernel_source.encode()).hexdigest()
 receipt=out/(str(seq)+'.dispatch.json')
 receipt.write_text(json.dumps(r,indent=2))
 try:
  result=old(k,*args,**kwargs);r['phase']='native_returned_before_sync';receipt.write_text(json.dumps(r,indent=2))
  torch.npu.synchronize();r['phase']='native_sync_completed';r['completed']=time.time();receipt.write_text(json.dumps(r,indent=2));return result
 except Exception as e:
  r['error']=str(e);receipt.write_text(json.dumps(r,indent=2));raise
JITKernel.__call__=trace
tla.PROVIDERS[op]=lambda case:factory(case,op,native,tilelang,out)
checker=P/'frozen/scripts/ascend_pair_check.py';sys.argv=[str(checker),'--op',op,'--case-id',cid,'--library',baseline,'--output',str(out/'checker.json')]
try:runpy.run_path(str(checker),run_name='__main__')
finally:
 try:
  if torch.npu.is_initialized():torch.npu.synchronize()
  (out/'final_sync.json').write_text(json.dumps(dict(success=True)))
 except Exception as e:
  (out/'final_sync.json').write_text(json.dumps(dict(success=False,error=str(e))))
  raise
