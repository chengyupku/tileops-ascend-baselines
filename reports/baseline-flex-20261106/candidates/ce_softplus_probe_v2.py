import sys,os,json,traceback
from pathlib import Path
R=Path('/opt/tileops-all-pairs/workspace');P=Path('/opt/baseline-flex-20261106/server2/ce-softplus-probe-v2');P.mkdir(exist_ok=True)
sys.path[:0]=[str(R/'TileOPs'),str(R/'TileOPs/src'),str(R),str(R/'reports/baseline-cache-20261027/python')]
sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
import _distutils_hack
_distutils_hack.add_shim()
r={}
try:
 import torch,torch_npu
 from benchmarks.ascend_all_pairs.tla import native
 tl=native.runtime();tl.set_cache_dir(str(P/'jit'))
 source='examples/cross_entropy_loss/example_cross_entro.py'
 m=native.load(source,'cross_entropy_fp32')
 k=m.cross_entropy_fp32(32,2,32,32,x_dtype='float32')
 x=torch.tensor([-1000.,-100.,-20.,-10.,-1.,0.,1.,10.,20.,21.,100.,1000.]+[.1]*20)
 a=torch.stack((torch.zeros_like(x),x),-1).npu(); y=torch.zeros(32,dtype=torch.int32,device='npu')
 out=k(a,y)[0];torch.npu.synchronize();v=out.cpu();ref=torch.nn.functional.softplus(x)
 r={'status':'probe_completed','max_abs':float((v-ref).abs().max()),'result':v.tolist(),'reference':ref.tolist()}
except Exception:r={'status':'failed','error':traceback.format_exc()}
(P/'RESULT.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
if r['status']=='failed':raise SystemExit(1)
