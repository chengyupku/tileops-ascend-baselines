"""Cache only constructor-defined inverse frequencies; all live-position rotation stays timed."""
import torch,types
from adapters.execution_boundary import ExecutionBoundary
from adapters.rope_pow_v10 import RopePower
from workloads.attention.gqa.rope import apply_dense_rope

def bind(case,args,previous):
 assert case._call.signature.name=='GQAPrefillPagedWithKVCacheFwdOp'
 w=case.workload;dim=w.dim if w.rotary_dim is None else w.rotary_dim;half=dim//2
 with RopePower():
  inv_freq=1.0/(w.rope_base**(torch.arange(half,device=args[0].device,dtype=torch.float32)/half))
 def rope(self,x,positions):
  angles=torch.outer(positions.float(),inv_freq)
  cos,sin=angles.cos().to(x.dtype),angles.sin().to(x.dtype)
  index=torch.arange(positions.numel(),device=x.device)
  return apply_dense_rope(x[None],index,cos,sin,rotary_dim=dim,layout='neox')[0]
 w.rope=types.MethodType(rope,w)
 return ExecutionBoundary(previous[0],tuple(args),previous[2],{'candidate':'gqa-constant-invfreq-v1','change':'constant base/dimension inverse frequency cached using identical FP64 power rounding; all positions, angles, cos/sin, attention and cache updates timed','input_dependent_work_timed':True})
