import json
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import sign
from tileops.ops.elementwise import SignFwdOp
rows=[]
for dtype,count in [(torch.int8,256),(torch.uint8,256),(torch.int16,65536)]:
 x=torch.arange(count,dtype=torch.int32).to(dtype);a=x.npu();actual=SignFwdOp()(a);expected=torch.sign(x)
 torch.testing.assert_close(actual.cpu(),expected,rtol=0,atol=0)
 try:torch.sign(a);torch.npu.synchronize()
 except RuntimeError as e:
  error=str(e);assert 'dtype' in error and 'support list' in error
 else:raise AssertionError('expected unsupported eager dtype; investigate changed backend')
 rows.append({'dtype':str(dtype),'all_patterns':count,'tileops_matches_cpu':True,'eager_error':error})
torch.npu.synchronize()
Path(__file__).with_name('sign-integer-diagnostic.json').write_text(json.dumps({'checks':rows,'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
print(json.dumps(rows),flush=True)
