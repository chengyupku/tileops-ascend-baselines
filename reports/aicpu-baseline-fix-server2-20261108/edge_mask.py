import sys,pathlib,json,hashlib,fcntl
P=pathlib.Path(__file__).resolve().parent
sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
with open('/run/lock/ascend-npu/device-0.lock') as f:
 try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:pass
 else:raise RuntimeError('guard required')
import torch,torch_npu
torch.npu.set_device(0)
patterns=torch.tensor([[(i>>b)&1 for b in range(8)] for i in range(256)],dtype=torch.uint8)
rows=[]
for truebyte in [1,2,127,128,255]:
 raw=patterns*truebyte;mask=raw.to('npu').view(torch.bool)
 bits=mask.to(torch.int32);weights=torch.tensor([1,2,4,8,16,32,64,128],dtype=torch.int32,device='npu')
 got=(bits*weights).sum(dim=-1,dtype=torch.int32).to(torch.uint8).cpu();want=torch.arange(256,dtype=torch.int32).to(torch.uint8)
 rows.append({'truebyte':truebyte,'all256_patterns_exact':torch.equal(got,want)})
torch.npu.synchronize();(P/'MASK_EDGE_CHECKS.json').write_text(json.dumps(rows,indent=2));print(rows,flush=True);assert all(r['all256_patterns_exact'] for r in rows)
