import os,sys,json,runpy,hashlib,ctypes
from pathlib import Path
P=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/threehost_linear49');S=P/'snapshot';W=P.parents[2];cid=sys.argv[1];D=Path(sys.argv[2])
assert any(x['baseline']=='flash-linear-attention-npu' and x['operator']=='GDNFwdOp' and x['case_id']==cid and x['status']=='blocked' for x in json.loads((P/'SCOPE.json').read_text()))
sys.path[:0]=[str(S/'TileOPs'),str(S/'TileOPs/src'),str(S),str(P)]
import benchmarks.ascend_baselines as base
base.WORKSPACE=W;base.LIBRARIES=W/'ascend_baselines'
from benchmarks.ascend_all_pairs import fla
from integration700_fla_native import build
fla.PROVIDERS['GDNFwdOp']=build
checker=runpy.run_path(str(P/'original_fla_gdn_checker.py'))
sys.path[:0]=[str(S/'TileOPs'),str(S/'TileOPs/src'),str(S),str(P)]
sys.argv=['checker','--op','GDNFwdOp','--case-id',cid,'--library','flash-linear-attention-npu','--output',str(D/'checker.json')]
try:rc=checker['main']()
finally:
 import torch
 if torch.npu.is_initialized():torch.npu.synchronize()
 (D/'maps.txt').write_text(Path('/proc/self/maps').read_text())
 libpath=Path(os.environ['ASCEND_CLASSIC_FLA_PACKAGE'])/'opp/vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so'
 lib=ctypes.CDLL(str(libpath));fn=getattr(lib,'aclnnChunkGatedDeltaRuleFwd');st=libpath.stat()
 (D/'NATIVE_IDENTITY.json').write_text(json.dumps(dict(path=str(libpath),sha256=hashlib.sha256(libpath.read_bytes()).hexdigest(),symbol='aclnnChunkGatedDeltaRuleFwd',pointer=hex(ctypes.cast(fn,ctypes.c_void_p).value),inode=st.st_ino,device=st.st_dev),indent=2))
 (D/'LOADED_PYTHON.json').write_text(json.dumps({n:getattr(m,'__file__',None) for n,m in sys.modules.items()},indent=2))
raise SystemExit(rc)
