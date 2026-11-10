import ctypes as C
import torch
from adapters.ops_math_binding import _MathCall

def bind(shape):
 binding=_MathCall('aclStft',[C.c_void_p]*3+[C.c_int64]*3+[C.c_bool]*3)
 binding.codes[torch.complex128]=17
 n=shape[-1]
 window=torch.ones((n,),dtype=torch.complex128,device='cpu').to('npu')
 def run(x):
  # Drain previous completed resources before retaining another complete invocation.
  binding.cleanup()
  out=torch.empty((*shape,1),dtype=x.dtype,device=x.device)
  record=binding.record()
  binding.call([binding.tensor(x,record),binding.tensor(window,record),binding.tensor(out,record),n,n,n,False,False,True],record,x.device)
  return out.reshape(shape)
 return run,binding.cleanup
