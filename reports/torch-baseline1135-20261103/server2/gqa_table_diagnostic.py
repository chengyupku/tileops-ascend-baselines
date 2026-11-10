import sys,pathlib,os,json,time
P=pathlib.Path('/opt/torch-baseline1135-20261103');sys.path[:0]=[str(P/'source/src'),str(P/'source'),str(P),'/opt/tileops-ci/runtime/lib/python3.12/site-packages'];os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import torch,torch_npu
from benchmarks.api import cases
from workloads.device import set_run_device
e=torch.arange(32,dtype=torch.float32)/32;ne=e.npu()
cp=10000.0**e;ci=1.0/cp
np=10000.0**ne;ni=1.0/np
variants={'original':ni,'tensor_pow':1.0/torch.pow(torch.full_like(ne,10000.),ne),'tensor_div':torch.ones_like(np)/np,'tensor_both':torch.ones_like(ne)/torch.pow(torch.full_like(ne,10000.),ne)}
r={}
for name,v in variants.items():
 a=v.cpu();r[name]={'inv_diff':int((a!=ci).sum()),'max_diff':float((a-ci).abs().max()),'values':a.tolist()}
r['cpu_inv']=ci.tolist();r['cpu_pow']=cp.tolist();r['npu_pow']=np.cpu().tolist();r['tensor_pow']=torch.pow(torch.full_like(ne,10000.),ne).cpu().tolist();r['double_inv']=(1.0/(10000.0**e.double())).float().tolist()
(P/'server2/GQA_TABLE_DIAGNOSTIC.json').write_text(json.dumps(r,indent=2));print(json.dumps(r));torch.npu.synchronize()
