from pathlib import Path
import os,sys,tempfile
def setup(P,W,V,C,R,out):
 for n in ['ASCEND_NPU_ARCH','TVM_HOME','TVM_IMPORT_PYTHON_PATH']:os.environ.pop(n,None)
 os.environ.update(TL_PLATFORM='A5',TL_ROOT=str(V),TL_TEMPLATE_PATH=str(V/'src'),TVM_IMPORT_PYTHON_PATH=str(V/'3rdparty/tvm/python'),TILELANG_LIBRARY_PATH=str(C/'handoff/lib'),TVM_LIBRARY_PATH=str(C/'handoff/lib'),SETUPTOOLS_USE_DISTUTILS='local',PYTHONDONTWRITEBYTECODE='1')
 for n in ['TILELANG_CACHE_DIR','TMPDIR','XDG_CACHE_HOME','TLA_CYTHON_CACHE_DIR','ASCEND_WORK_PATH','ASCEND_CACHE_PATH','ASCEND_PROCESS_LOG_PATH','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR']:
  d=out/'cache'/n;d.mkdir(parents=True,exist_ok=True);os.environ[n]=str(d)
 tempfile.tempdir=os.environ['TMPDIR'];sys.dont_write_bytecode=True
 os.environ['PATH']=str(C/'handoff/bin')+':'+os.environ['PATH']
 sys.path[:0]=[str(R),str(V/'3rdparty/tvm/python'),str(C/'python'),str(W/'.cache/all-pairs/remote_native_tla/python-deps'),str(P),str(P/'frozen'),str(P/'frozen/src'),str(W/'TileOPs/src'),str(W)]
 sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
 sdk=W/'.cache/all-pairs/remote_native_deepgemm_deepep/sdk-920b2'
 os.environ.update(ASCEND_HOME_PATH=str(sdk),ASCEND_TOOLKIT_HOME=str(sdk),BISHENG_HOME=str(sdk/'tools/bisheng_compiler'))
 os.environ['PATH']=str(sdk/'tools/bisheng_compiler/bin')+':'+os.environ['PATH']
