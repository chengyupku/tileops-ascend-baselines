import pathlib,sys,json,fcntl
P=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(P),'/opt/tileops-ci/runtime/lib/python3.12/site-packages']
with open('/run/lock/ascend-npu/device-0.lock') as f:
 try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:pass
 else:raise RuntimeError('guard required')
import torch,torch_npu
torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'})
rows={}
w=torch.tensor([[1.,2**-25,2**-25,2**-25]],dtype=torch.float32,device='npu')
wide=w.double();cdf=wide.cumsum(-1)
rows['cdf']={'wide_dtype':str(wide.dtype),'cdf_dtype':str(cdf.dtype),'actual':cdf.cpu().tolist(),'cpu_fp64':w.cpu().double().cumsum(-1).tolist()}
# Build representable FP64 intermediates on device, not via double CPU->NPU conversion.
a=torch.ones((1,4),device='npu',dtype=torch.float64)
a=a+torch.tensor([[0.,2**-25,2**-24,2**-23]],device='npu',dtype=torch.float32)
point=torch.ones((1,1),device='npu',dtype=torch.float64)+torch.tensor(2**-26,device='npu')
old=torch.searchsorted(a,point,right=True)
new=(a<=point).sum(-1,keepdim=True)
rows['close_boundary']={'a_dtype':str(a.dtype),'a':a.cpu().tolist(),'point':point.cpu().tolist(),'searchsorted':old.cpu().tolist(),'comparison_count':new.cpu().tolist(),'cpu_searchsorted':torch.searchsorted(a.cpu(),point.cpu(),right=True).tolist()}
torch.npu.synchronize();(P/'SAMPLING_PRECISION_PROBE.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows))
