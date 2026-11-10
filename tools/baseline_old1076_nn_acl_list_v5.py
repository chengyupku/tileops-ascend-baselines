"""Original ACLNN TensorList construction, following vendor public API example."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
os.environ['ASCEND_CLASSIC_NN_VENDOR']=os.environ['OPS_NN_VENDOR']
source=(ROOT/'tools/baseline_old1076_nn_arm_staging_v3.py').read_text()
insert='''
def direct_list(case):
 import ctypes as C
 import torch
 from benchmarks.api import Implementation
 from benchmarks.ascend_baselines import Candidate,LIBRARIES,source_hashes
 from benchmarks.ascend_math_more import _MathCall
 from benchmarks.ascend_all_pairs.ops_nn.catalog import FOREACH
 api,directory=FOREACH[case._call.signature.name]
 assert api in ('ForeachMulList','ForeachMinimumList')
 binding=_MathCall('aclnn'+api,[C.c_void_p]*3,library='ops-nn')
 binding.base.aclCreateTensorList.argtypes=[C.POINTER(C.c_void_p),C.c_uint64]
 binding.base.aclCreateTensorList.restype=C.c_void_p
 binding.base.aclDestroyTensorList.argtypes=[C.c_void_p]
 binding.base.aclDestroyTensorList.restype=C.c_int
 lists=[]
 def cleanup():
  torch.npu.synchronize()
  while lists:binding.check(binding.base.aclDestroyTensorList(lists.pop()),'aclDestroyTensorList')
  binding.cleanup()
 def run(x,y):
  cleanup()
  shape=torch.broadcast_shapes(x.shape,y.shape)
  xx=x.expand(shape).contiguous();yy=y.expand(shape).contiguous();out=torch.empty(shape,dtype=x.dtype,device=x.device)
  record=binding.record();arguments=[]
  for t in (xx,yy,out):
   tensor=binding.tensor(t,record);storage=(C.c_void_p*1)(tensor);record['keep'].append(storage)
   ptr=binding.base.aclCreateTensorList(storage,1)
   if not ptr:raise RuntimeError('aclCreateTensorList returned null')
   # Original example transfers tensor descriptor destruction to TensorList.
   record['tensors'].remove(tensor);lists.append(ptr);arguments.append(ptr)
  binding.call(arguments,record,x.device)
  return out
 return Candidate(Implementation(run),'ops-nn::aclnn'+api,'Original public ACLNN two-stage API with original aclCreateTensorList and descriptor ownership; per-call broadcast materialization; same original vendor kernel/compiler and full checker.',binding.hashes(LIBRARIES/'ops-nn'/directory)|source_hashes(Path(__file__)),cleanup=cleanup)
'''
source=source.replace('def factory(case,library):',insert+'\ndef factory(case,library):').replace("c=staging.CANDIDATES[case._call.signature.name](case)",'c=direct_list(case)')
exec(compile(source,str(Path(__file__)),'exec'))
