"""Final exhaustive 16-bit and dense FP32 transition tests; independent CPU truth."""
import hashlib,json,sys
from pathlib import Path
output=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).with_name('audit')/'unary-final-numerical.json'
assert not output.exists(), output
ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src')]
sys.path.insert(0,str(ROOT/'docs/ascend950/dev/lerp_exp4_20261028'))
import torch,torch_npu
from forbid_events import install
install(torch,torch_npu)
from tileops.kernels.ascend.elementwise import exp,exponential
from tileops.ops.elementwise import ExpFwdOp
from workloads.device import set_run_device
from workloads.numerics import compare_outputs,Exact
from full_digest import tensor_digest
set_run_device('npu');torch.npu.set_device(0)
rows=[]
for cls,reference in [(ExpFwdOp,torch.exp)]:
 for dt in [torch.float16,torch.bfloat16,torch.float32]:
  if dt!=torch.float32:
   cpu=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(dt)
  else:
   boundary=torch.tensor([-1/32,1/32],dtype=dt)
   cpu=torch.cat([torch.linspace(-.04,.04,65535),torch.linspace(-1e-6,1e-6,8193),boundary,
      torch.nextafter(boundary,torch.full_like(boundary,float('inf'))),torch.nextafter(boundary,torch.full_like(boundary,float('-inf'))),
      torch.tensor([0.,-0.,1e-30,-1e-30,88.,89.,float('inf'),float('-inf'),float('nan')])])
  x=cpu.to('npu');before=tensor_digest([x]);op=cls();got=op(x);expected=reference(cpu)
  check=compare_outputs(got.cpu(),expected,Exact());assert check.checked_outputs
  assert tensor_digest([x])==before and got.data_ptr()!=x.data_ptr()
  finite=cpu.isfinite(); changed_cpu=cpu.clone();changed_cpu[finite]=torch.clamp(changed_cpu[finite].float(),-.01,.01).to(dt)
  changed=changed_cpu.to('npu');second=op(changed)
  compare_outputs(second.cpu(),reference(changed_cpu),Exact())
  try:compare_outputs(torch.zeros_like(expected),expected,Exact())
  except AssertionError:pass
  else:raise AssertionError('comparator accepted zeroed output')
  row=dict(op=cls.__name__,dtype=str(dt),n=cpu.numel(),correct=True,max_abs_err=check.max_abs_err,
           input_unchanged=True,changed_input_checked=True,negative_control_rejected=True)
  print(json.dumps(row),flush=True);rows.append(row)
torch.npu.synchronize()
files=[Path(__file__),Path(exponential.__file__),Path(exp.__file__),ROOT/'docs/ascend950/dev/lerp_exp4_20261028/full_digest.py',ROOT/'docs/ascend950/dev/lerp_exp4_20261028/forbid_events.py',ROOT/'TileOPs/workloads/numerics.py']
output.write_text(json.dumps(dict(runtime_limit=torch.npu.get_device_limit(0),rows=rows,sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}),indent=2))
