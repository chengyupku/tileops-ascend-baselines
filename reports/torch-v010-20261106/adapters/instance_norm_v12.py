"""Current CPU native InstanceNorm FP16 saved-stat rounding, computed on NPU."""
import torch
def reference(x,running_mean,running_var,weight,bias,*,eps):
 assert x.dtype==torch.float16 and running_mean is None and running_var is None
 xf=x.float();dims=tuple(range(2,x.ndim));mean=xf.mean(dims,keepdim=True)
 var=((xf-mean)**2).mean(dims,keepdim=True)
 # CPU native_batch_norm returns saved mean and invstd in input dtype.
 mean=mean.to(x.dtype).float();invstd=torch.rsqrt(var+eps).to(x.dtype).float()
 shape=[1,-1]+[1]*(x.ndim-2)
 alpha=invstd if weight is None else invstd*weight.float().reshape(shape)
 beta=-mean*alpha
 if bias is not None:beta=bias.float().reshape(shape)+beta
 return (xf*alpha+beta).to(x.dtype)
