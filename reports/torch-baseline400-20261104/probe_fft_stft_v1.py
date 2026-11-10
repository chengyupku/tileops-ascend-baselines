import sys,os,json,traceback
sys.path.insert(0,'/opt/tileops-ci/runtime/lib/python3.12/site-packages')
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import torch,torch_npu
torch.set_num_threads(8);torch.manual_seed(42)
rows=[]
for shape in [(64,4096),(4,1048576)]:
 try:
  x=torch.randn(shape,dtype=torch.complex128);expected=torch.fft.fft(x,dim=-1);n=shape[-1]
  xn=x.npu();window=torch.ones(n,dtype=torch.complex128).npu()
  y=torch.stft(xn,n_fft=n,hop_length=n,win_length=n,window=window,center=False,normalized=False,onesided=False,return_complex=True).reshape(shape)
  torch.npu.synchronize();actual=y.cpu();atol=1e-8 if n==4096 else 2e-11
  rows.append({'shape':shape,'dtype':str(actual.dtype),'max_abs_err':(actual-expected).abs().max().item(),'allclose':torch.allclose(actual,expected,atol=atol,rtol=1e-8)})
 except Exception as e:rows.append({'shape':shape,'error':str(e),'traceback':traceback.format_exc()});break
print(json.dumps(rows,indent=2),flush=True)
