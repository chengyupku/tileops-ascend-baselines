import sys,os,json,ctypes,traceback
sys.path.insert(0,'/opt/tileops-ci/runtime/lib/python3.12/site-packages')
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
result={}
try:
 ctypes.CDLL('/usr/local/Ascend/nnal/asdsip/9.1.0/lib/libasdsip.so');result['library_loaded']=True
 import torch,torch_npu
 torch.manual_seed(42)
 for dtype in [torch.complex64,torch.complex128]:
  try:
   x=torch.randn(4096,dtype=dtype);expected=torch.fft.fft(x);y=torch.fft.fft(x.npu());torch.npu.synchronize();actual=y.cpu()
   result[str(dtype)]={'output_dtype':str(actual.dtype),'max_abs_err':(actual-expected).abs().max().item(),'allclose_1e8':torch.allclose(actual,expected,rtol=1e-8,atol=1e-8)}
  except Exception as e:result[str(dtype)]={'error':str(e),'traceback':traceback.format_exc()}
except Exception as e:result['error']=str(e)
print(json.dumps(result,indent=2),flush=True)
