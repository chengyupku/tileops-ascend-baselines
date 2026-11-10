from pathlib import Path
import sys,os,json,runpy,time,gc
P=Path(__file__).resolve().parent;W=Path('/opt/tileops-all-pairs/workspace')
batch=json.loads((P/sys.argv[1]).read_text())
meta=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
assert meta['phase']=='running' and meta['owner']=='baseline700-20261026/tilelang_register5' and meta['child']['pid']==os.getpgrp()
for n in ['TILELANG_LIBRARY_PATH','TVM_LIBRARY_PATH','TVM_HOME','TVM_IMPORT_PYTHON_PATH','TL_ROOT','TL_PLATFORM']:os.environ.pop(n,None)
for n in ['TILELANG_CACHE_DIR','TMPDIR','XDG_CACHE_HOME','ASCEND_WORK_PATH','ASCEND_CACHE_PATH','ASCEND_PROCESS_LOG_PATH','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR']:
 d=P/'tilelang_cache'/n;d.mkdir(parents=True,exist_ok=True);os.environ[n]=str(d)
sys.path[:0]=[str(P),str(W/'TileOPs'),str(W/'TileOPs/src'),str(W)]
sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
sdk=W/'.cache/all-pairs/remote_native_deepgemm_deepep/sdk-920b2'
os.environ.update(BISHENG_HOME=str(sdk/'tools/bisheng_compiler'),ASCEND_HOME_PATH=str(sdk),ASCEND_TOOLKIT_HOME=str(sdk),CPATH=':'.join(str(sdk/x) for x in ['asc/include','asc','include']))
import torch,tilelang
assert Path(tilelang.__file__).resolve()==Path('/opt/tileops-ci/runtime/lib/python3.12/site-packages/tilelang/__init__.py')
from benchmarks.ascend_all_pairs import tilelang as provider
for op,cid in batch:
 out=P/'runs'/('tilelang_'+op+'_'+cid);out.mkdir(parents=True,exist_ok=True);os.chdir(out)
 (out/'GUARD_RUNNING.json').write_text(json.dumps(meta,indent=2))
 checker=W/'TileOPs/scripts/ascend_pair_check.py';sys.argv=[str(checker),'--op',op,'--case-id',cid,'--library','tilelang','--output',str(out/'checker.json')]
 start=time.time();rc=1
 try:runpy.run_path(str(checker),run_name='__main__')
 except SystemExit as e:rc=e.code or 0
 finally:
  try:
   if torch.npu.is_initialized():torch.npu.synchronize()
  except Exception:rc=1
 (out/'guard.json').write_text(json.dumps(dict(key=['tilelang',op,cid],started=start,finished=time.time(),returncode=rc,shared_guard=True),indent=2))
 gc.collect()
 if rc:break
