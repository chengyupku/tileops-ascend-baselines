import sys,pathlib,json,hashlib,fcntl
P=pathlib.Path(__file__).resolve().parent;B=pathlib.Path('/opt/torch-all2208-20261105')
sys.path[:0]=[str(P),str(B/'source'),str(B/'source/src'),'/opt/tileops-ci/runtime/lib/python3.12/site-packages']
with open('/run/lock/ascend-npu/device-0.lock') as f:
 try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:pass
 else:raise RuntimeError('host guard required')
import torch,torch_npu
from workloads.quantization.quantize import int4_quant_per_group as original
from fixes import int4_quant_per_group as fixed
torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});torch.set_num_threads(8)
q=torch.tensor([(a,b) for a in range(-8,8) for b in range(-8,8)],dtype=torch.float32)
expected=(q[:,0].to(torch.int8)<<4)|(q[:,1].to(torch.int8)&15)
d=q.to('npu');low=d[:,1];got=(d[:,0]*16+torch.where(low<0,low+16,low)).to(torch.int8).cpu();assert torch.equal(expected,got)
rows=[];torch.manual_seed(1234)
for shape,group in [((32,64),64),((32,128),128),((32,4096),4096),((4096,4096),64)]:
 cpu=torch.randn(shape,dtype=torch.float16);cpu[0].zero_();cpu[1].fill_(2);x=cpu.to('npu');old=original(x,group);new=fixed(x,group)
 assert all(torch.equal(a.cpu(),b.cpu()) for a,b in zip(old,new))
 rows.append({'shape':shape,'group':group,'packed_and_params_exact':True,'zero_and_constant_rows':True})
torch.npu.synchronize();(P/'INT4_EDGE_CHECKS.json').write_text(json.dumps({'code_pairs_exact':256,'checks':rows,'fixes_sha256':hashlib.sha256((P/'fixes.py').read_bytes()).hexdigest()},indent=2));print('INT4 EDGE PASS',flush=True)
