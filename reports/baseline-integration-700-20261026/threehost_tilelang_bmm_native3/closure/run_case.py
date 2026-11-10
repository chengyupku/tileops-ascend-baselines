from pathlib import Path
import os,sys,json,runpy,time,hashlib,ctypes,inspect
P=Path(__file__).resolve().parent;W=Path('/opt/tileops-all-pairs/workspace')
baseline,op,cid=sys.argv[1:4]
assert [baseline,op,cid] in [[r['baseline'],r['operator'],r['case_id']] for r in json.loads((P/'SCOPE.json').read_text())]
meta=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert meta['owner']=='baseline700-20261026/threehost_tilelang_bmm_native3' and meta['phase']=='running' and meta['child']['pid']==os.getpgrp()
out=P/'runs'/('_'.join([baseline,op,cid])+(sys.argv[4] if len(sys.argv)>4 else ''));out.mkdir(parents=True,exist_ok=False);os.chdir(out)
(out/'GUARD_RUNNING.json').write_text(json.dumps(meta,indent=2))
for n in ['TILELANG_LIBRARY_PATH','TVM_LIBRARY_PATH','TVM_HOME','TVM_IMPORT_PYTHON_PATH','TL_ROOT','TL_PLATFORM','ASCEND_NPU_ARCH']:os.environ.pop(n,None)
for n in ['TILELANG_CACHE_DIR','TMPDIR','XDG_CACHE_HOME','ASCEND_WORK_PATH','ASCEND_CACHE_PATH','ASCEND_PROCESS_LOG_PATH','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR']:
 d=P/'cache'/n;d.mkdir(parents=True,exist_ok=True);os.environ[n]=str(d)
import tempfile;tempfile.tempdir=os.environ['TMPDIR']
sys.path[:0]=[str(P/'frozen'),str(P),str(W/'TileOPs/src'),str(W)]
sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
sdk=W/'.cache/all-pairs/remote_native_deepgemm_deepep/sdk-920-release-headers'
os.environ.update(BISHENG_HOME=str(sdk/'tools/bisheng_compiler'),ASCEND_HOME_PATH=str(sdk),ASCEND_TOOLKIT_HOME=str(sdk),CPATH=':'.join(str(sdk/x) for x in ['asc/include','asc','include']))
import torch,tilelang
from benchmarks.ascend_all_pairs import tilelang as provider
assert str(P/'frozen') in provider.__file__
from tilelang_adapter import factory
from tilelang.jit.kernel import JITKernel
import tvm
load_module_original=tvm.runtime.load_module
load_sequence=0
def load_module_traced(path,*args,**kwargs):
 global load_sequence
 f=Path(path);data=f.read_bytes();st=f.stat();sha=hashlib.sha256(data).hexdigest()
 dest=out/'native_artifacts'/sha;dest.parent.mkdir(exist_ok=True);dest.write_bytes(data)
 mod=load_module_original(path,*args,**kwargs)
 record=dict(path=str(f),sha256=sha,inode=st.st_ino,device=st.st_dev,symbols={},maps=Path('/proc/self/maps').read_text())
 try:
  handle=ctypes.CDLL(str(f),mode=os.RTLD_NOLOAD|os.RTLD_LOCAL)
  for symbol in ['__tvm_ffi_main','main']:
   try:record['symbols'][symbol]=ctypes.cast(getattr(handle,symbol),ctypes.c_void_p).value
   except Exception:pass
 except Exception as e:record['error']=str(e)
 load_sequence+=1;(out/('native_load_'+str(load_sequence)+'.json')).write_text(json.dumps(record,indent=2))
 return mod
tvm.runtime.load_module=load_module_traced
original=JITKernel.__call__;seq=0
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def trace(k,*args,**kwargs):
 global seq
 seq+=1;ad=k.adapter
 r=dict(sequence=seq,time=time.time(),target=str(k.target),adapter=str(type(ad)),execution_backend=k.execution_backend,inputs=[dict(shape=list(x.shape),dtype=str(x.dtype),device=str(x.device),numel=x.numel()) for x in args if isinstance(x,torch.Tensor)],symbols={},maps=Path('/proc/self/maps').read_text())
 lib=getattr(ad,'lib',None);r['library']=getattr(lib,'_name',None)
 for name in ['call','init','kernel','launch']:
  try:r['symbols'][name]=ctypes.cast(getattr(lib,name),ctypes.c_void_p).value
  except Exception:pass
 if r['library'] and Path(r['library']).is_file():
  f=Path(r['library']);st=f.stat();r.update(binary_sha256=digest(f),inode=st.st_ino,device=st.st_dev); (out/(str(seq)+'.kernel.txt')).write_text(k.get_kernel_source())
 (out/(str(seq)+'.dispatch.json')).write_text(json.dumps(r,indent=2))
 ans=original(k,*args,**kwargs);torch.npu.synchronize()
 r['maps_after']=Path('/proc/self/maps').read_text();r['loaded_native']=[]
 for path in sorted({line.split()[-1] for line in r['maps_after'].splitlines() if str(P) in line and '.so' in line}):
  f=Path(path)
  if not f.is_file():continue
  st=f.stat();nr=dict(path=path,sha256=digest(f),inode=st.st_ino,device=st.st_dev,symbols={})
  try:
   handle=ctypes.CDLL(path,mode=os.RTLD_NOLOAD|os.RTLD_LOCAL)
   for symbol in ['main','__tvm_ffi_main','__tvm_main__']:
    try:nr['symbols'][symbol]=ctypes.cast(getattr(handle,symbol),ctypes.c_void_p).value
    except Exception:pass
  except Exception as e:nr['symbol_error']=str(e)
  dest=out/'native_artifacts'/nr['sha256'];dest.parent.mkdir(exist_ok=True)
  if not dest.exists():dest.write_bytes(f.read_bytes())
  r['loaded_native'].append(nr)
 r['kernel_source_sha256']=hashlib.sha256(k.get_kernel_source().encode()).hexdigest();(out/(str(seq)+'.kernel.txt')).write_text(k.get_kernel_source())
 r['completed']=time.time();r['outputs']=[dict(shape=list(x.shape),dtype=str(x.dtype),numel=x.numel()) for x in (ans if isinstance(ans,(list,tuple)) else [ans]) if isinstance(x,torch.Tensor)];(out/(str(seq)+'.dispatch.json')).write_text(json.dumps(r,indent=2));return ans
JITKernel.__call__=trace
provider.PROVIDERS[op]=lambda case:factory(case,op,out)
checker=P/'frozen/scripts/ascend_pair_check.py';sys.argv=[str(checker),'--op',op,'--case-id',cid,'--library',baseline,'--output',str(out/'checker.json')]
t=time.time()
try:runpy.run_path(str(checker),run_name='__main__')
finally:
 if torch.npu.is_initialized():torch.npu.synchronize()
 (out/'completion.json').write_text(json.dumps(dict(wall_seconds=time.time()-t,checker_sha256=digest(checker)),indent=2))
