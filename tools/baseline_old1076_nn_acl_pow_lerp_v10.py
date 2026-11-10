"""Final bounded ACLNN dequant interface; original native multiply unchanged."""
import os,sys,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['ASCEND_CLASSIC_NN_VENDOR']=os.environ['OPS_NN_VENDOR']
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
from benchmarks.ascend_all_pairs import registry

def direct_list(case):
 import ctypes as C
 import torch
 from benchmarks.api import Implementation
 from benchmarks.ascend_baselines import Candidate,LIBRARIES,source_hashes
 from benchmarks.ascend_math_more import _MathCall
 is_lerp=case._call.signature.name=='LerpScalarFwdOp'
 api,directory=('ForeachLerpScalar','foreach/foreach_lerp_scalar') if is_lerp else ('ForeachPowList','foreach/foreach_pow_list')
 assert api in ('ForeachLerpScalar','ForeachPowList')
 binding=_MathCall('aclnn'+api,[C.c_void_p]*(4 if is_lerp else 3),library='ops-nn')
 binding.base.aclCreateTensorList.argtypes=[C.POINTER(C.c_void_p),C.c_uint64]
 binding.base.aclCreateTensorList.restype=C.c_void_p
 binding.base.aclDestroyTensorList.argtypes=[C.c_void_p]
 binding.base.aclDestroyTensorList.restype=C.c_int
 lists=[];scalars=[]
 binding.base.aclCreateScalar.argtypes=[C.c_void_p,C.c_int];binding.base.aclCreateScalar.restype=C.c_void_p
 binding.base.aclDestroyScalar.argtypes=[C.c_void_p];binding.base.aclDestroyScalar.restype=C.c_int
 def cleanup():
  torch.npu.synchronize()
  while lists:binding.check(binding.base.aclDestroyTensorList(lists.pop()),'aclDestroyTensorList')
  while scalars:binding.check(binding.base.aclDestroyScalar(scalars.pop()),'aclDestroyScalar')
  binding.cleanup()
 def run(x,y):
  cleanup()
  dequant=case._call.signature.name.startswith("INT8Dequant")
  if dequant:
   m,k=x.shape
   x=x.to(torch.float32)
   y=y.expand(m,k) if "PerTensor" in case._call.signature.name else y.repeat_interleave(128,dim=1)[:,:k]
  shape=torch.broadcast_shapes(x.shape,y.shape)
  xx=x.expand(shape).contiguous();yy=y.expand(shape).contiguous();out=torch.empty(shape,dtype=x.dtype,device=x.device)
  record=binding.record();arguments=[]
  for t in (xx,yy,out):
   tensor=binding.tensor(t,record);storage=(C.c_void_p*1)(tensor);record['keep'].append(storage)
   ptr=binding.base.aclCreateTensorList(storage,1)
   if not ptr:raise RuntimeError('aclCreateTensorList returned null')
   # Original example transfers tensor descriptor destruction to TensorList.
   record['tensors'].remove(tensor);lists.append(ptr);arguments.append(ptr)
  if is_lerp:
   weight=C.c_float(float(case.params['weight']));record['keep'].append(weight)
   scalar=binding.base.aclCreateScalar(C.byref(weight),0)
   if not scalar:raise RuntimeError('aclCreateScalar failed')
   scalars.append(scalar);arguments.insert(2,scalar)
  binding.call(arguments,record,x.device)
  return out.to(getattr(torch,case.params["out_dtype"])) if dequant else out
 return Candidate(Implementation(run),'ops-nn::aclnn'+api,'Original public ACLNN two-stage API with original aclCreateTensorList and descriptor ownership; per-call broadcast materialization; same original vendor kernel/compiler and full checker.',binding.hashes(LIBRARIES/'ops-nn'/directory)|source_hashes(Path(__file__)),cleanup=cleanup)

def factory(case,library):
 if library!='ops-nn' or case._call.signature.name not in ('PowFwdOp','LerpScalarFwdOp'):
  raise ValueError('Only reviewed Pow and scalar Lerp original APIs')
 return lambda:direct_list(case)
registry.candidate_factory=factory
runpy.run_path(str(ROOT/'tools/check_baseline_old1076_pair.py'),run_name='__main__')
