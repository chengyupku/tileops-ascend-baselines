"""Exact FP32 rounding across exponent ranges and neighbors of integer boundaries."""
import json,hashlib
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import ceil as module
from tileops.ops.elementwise import CeilFwdOp
torch.manual_seed(20261030)
bits=torch.randint(0,2**32,(65536,),dtype=torch.int64).to(torch.int32)
base=torch.tensor([-8388609.,-8388608.,-8388607.,-2.,-1.,-.5,-0.,0.,.5,1.,2.,8388607.,8388608.,8388609.],dtype=torch.float32)
values=torch.cat([bits.view(torch.float32),base,torch.nextafter(base,torch.full_like(base,float('inf'))),torch.nextafter(base,torch.full_like(base,-float('inf')))])
a=values.npu();op=CeilFwdOp();actual=op(a).cpu();expected=torch.ceil(values.double()).float()
torch.testing.assert_close(actual,expected,rtol=0,atol=0,equal_nan=True)
finite=~torch.isnan(expected);assert torch.equal(torch.signbit(actual[finite]),torch.signbit(expected[finite]))
assert torch.equal(a.cpu().view(torch.uint8),values.view(torch.uint8))
changed=torch.linspace(-1000,1000,len(values),dtype=torch.float32).npu();second=op(changed).cpu()
torch.testing.assert_close(second,torch.ceil(changed.cpu().double()).float(),rtol=0,atol=0)
bad=actual.clone();bad[65536]=123
try:torch.testing.assert_close(bad,expected,rtol=0,atol=0,equal_nan=True)
except AssertionError:pass
else:raise AssertionError('negative control accepted')
torch.npu.synchronize()
Path(__file__).with_name('ceil-fp32-edges.json').write_text(json.dumps({'values':len(values),'passed':True,'changed_input':True,'negative_control_rejected':True,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
