"""Original ACLNN TensorList construction, following vendor public API example."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
os.environ['ASCEND_CLASSIC_NN_VENDOR']=os.environ['OPS_NN_VENDOR']
import ctypes
_cann_host=ctypes.CDLL('/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64/libopapi.so',mode=os.RTLD_NOW|os.RTLD_GLOBAL)
_subset=ctypes.CDLL(str(Path(os.environ['OPS_NN_VENDOR'])/'op_api/lib/libcust_opapi.so'),mode=os.RTLD_NOW|os.RTLD_LOCAL|os.RTLD_DEEPBIND)
source='"""Validate original staged NN candidates under the unchanged full checker."""\nimport dataclasses,runpy,sys,types\nfrom pathlib import Path\nROOT=Path(__file__).resolve().parents[1]\nsys.path[:0]=[str(ROOT/\'TileOPs\'),str(ROOT/\'TileOPs/src\'),str(ROOT)]\nimport ctypes,os\nvendor=Path(os.environ[\'OPS_NN_VENDOR\'])\nctypes.CDLL(\'/usr/local/Ascend/cann-9.1.0/lib64/libes_nn.so\',mode=ctypes.RTLD_GLOBAL)\nctypes.CDLL(str(vendor/\'op_proto/es/lib/linux/aarch64/libes_nn.so\'),mode=ctypes.RTLD_GLOBAL)\nfrom benchmarks.ascend_all_pairs import registry\nname=\'benchmarks.ascend_all_pairs.ops_nn\'\npackage=types.ModuleType(name);package.__path__=[str(ROOT/\'reports/baseline-old1076-20261029/arm_nn_adapters\')];package.__package__=name\nsys.modules[name]=package\nfrom benchmarks.ascend_all_pairs.ops_nn import staging\nfrom benchmarks.ascend_baselines import source_hashes\n\ndef factory(case,library):\n if library!=\'ops-nn\':raise ValueError(\'ops-nn staging only\')\n def build():\n  c=staging.CANDIDATES[case._call.signature.name](case)\n  return dataclasses.replace(c,source_sha256={**c.source_sha256,**source_hashes(Path(__file__))})\n return build\nregistry.candidate_factory=factory\nrunpy.run_path(str(ROOT/\'tools/measure_baseline_cache_pair.py\'),run_name=\'__main__\')\n'
insert='''
def direct_list(case):
 import ctypes as C
 import torch
 from benchmarks.api import Implementation
 from benchmarks.ascend_baselines import Candidate,LIBRARIES,source_hashes
 from benchmarks.ascend_math_more import _MathCall
 from benchmarks.ascend_all_pairs.ops_nn.catalog import FOREACH
 api,directory=FOREACH[case._call.signature.name]
 assert api in ('ForeachMulList','ForeachMinimumList','ForeachMaximumList','ForeachDivList','ForeachPowList')
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
  binding.call(arguments,record,x.device)
  return out.to(getattr(torch,case.params["out_dtype"])) if dequant else out
 return Candidate(Implementation(run),'ops-nn::aclnn'+api,'Original public ACLNN two-stage API with original aclCreateTensorList and descriptor ownership; per-call broadcast materialization; same original vendor kernel/compiler and full checker.',binding.hashes(LIBRARIES/'ops-nn'/directory)|source_hashes(Path(__file__)),cleanup=cleanup)
'''
source=source.replace('def factory(case,library):',insert+'\ndef factory(case,library):').replace("c=staging.CANDIDATES[case._call.signature.name](case)",'c=direct_list(case)')
exec(compile(source,str(Path(__file__)),'exec'))
