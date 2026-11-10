"""Controlled full-call comparison under the 32 AIC / 64 AIV runtime limit."""
import hashlib
import json
from pathlib import Path
import torch
import torch_npu
from tileops.kernels.ascend.elementwise import bitwise_binary as control
from tileops.ops.elementwise import BitwiseAndFwdOp
from tileops.backend import register_kernel_type
from tools.measure_baselines import _time_reference
from benchmarks.api import cases
from workloads.device import set_run_device
from tools.baseline_protocol import seed_case
import bool_candidate
HERE=Path(__file__).resolve().parent
assert torch.npu.get_device_limit(0)=={'cube_core_num':32,'vector_core_num':64}
torch.npu.Event=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('Event forbidden'))
set_run_device('npu')
case=next(c for c in cases('BitwiseAndFwdOp') if c.id=='hidden-state-prefill-bool')
seed_case(torch,'BitwiseAndFwdOp',case.id)
xs=case.inputs
reference=case.reference(*(x.cpu() for x in xs))
old=BitwiseAndFwdOp()
torch.testing.assert_close(old(*xs).cpu(),reference,rtol=0,atol=0)
old_profile=_time_reference(torch,torch_npu,old,tuple(xs),3,15)
bool_candidate.AscendBitwiseAndKernel.preferred_over=frozenset({'ascend'})
register_kernel_type('BitwiseAndFwdOp','ascend_bool_candidate',bool_candidate.AscendBitwiseAndKernel)
new=BitwiseAndFwdOp()
torch.testing.assert_close(new(*xs).cpu(),reference,rtol=0,atol=0)
# All 256 storage bytes, every pair: semantic bool AND must produce canonical bytes.
a=torch.arange(256,dtype=torch.uint8).repeat_interleave(256)
b=torch.arange(256,dtype=torch.uint8).repeat(256)
for n in [1,7,129,65536]:
 da=a[:n].npu().view(torch.bool);db=b[:n].npu().view(torch.bool)
 got=BitwiseAndFwdOp()(da,db)
 expected=torch.logical_and(a[:n]!=0,b[:n]!=0)
 torch.testing.assert_close(got.cpu(),expected,rtol=0,atol=0)
 torch.testing.assert_close(got.view(torch.uint8).cpu(),expected.to(torch.uint8),rtol=0,atol=0)
profile=_time_reference(torch,torch_npu,new,tuple(xs),3,15)
eager=_time_reference(torch,torch_npu,case.reference,tuple(xs),3,15)
built=[v for group in new._built_entries.values() for v in group.values()]
assert len(built)==1 and type(built[0]).__module__=='bool_candidate'
(HERE/'bool-candidate.c').write_text(built[0].compiled.get_kernel_source())
result={'case':case.id,'control':old_profile,'candidate':profile,'eager':eager,'runtime_limit':torch.npu.get_device_limit(0),'correct':True,'raw_bool_checks':[1,7,129,65536],'scope':'candidate experiment, not full manifest acceptance','sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(control.__file__),Path(bool_candidate.__file__),Path(__file__)]}}
(HERE/'bool-probe.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
torch.npu.synchronize()
