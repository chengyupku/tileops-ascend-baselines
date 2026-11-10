"""CPU-only positive/negative controls for the proposed eager checker correction."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

BASE = Path('/opt/tileops-ci/server2-20261030/overnight-20261101')
sys.path.insert(0, str(BASE/'ci-full-v3/workspace/TileOPs'))
import torch
from workloads.numerics import Custom, Exact

assert 'torch_npu' not in sys.modules and not hasattr(torch, 'npu')
spec = importlib.util.spec_from_file_location('prepared_eager', BASE/'PREPARED_elementwise_eager_check.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
torch_proxy = SimpleNamespace(Tensor=torch.Tensor, npu=SimpleNamespace(synchronize=lambda:None))
x = torch.tensor([1., 2., 3.])
calls = []


class Workload:
    def verification(self, supplied):
        calls.append(supplied)

        def validate(got, expected):
            torch.testing.assert_close(got, supplied*2, atol=0, rtol=0)

        return Custom(validate, 'same scaling policy')


w = Workload()
original = w.verification(x)
calls.clear()
case = SimpleNamespace(reference=lambda a:a*2, verification=original, workload=w)
assert helper.check_eager(torch_proxy, case, (x,), x*2)['eligible']
assert len(calls) == 1 and calls[0] is not x
assert torch.equal(calls[0], x)
for wrong in (lambda a:a*3, lambda a:(a*2,a*2)):
    case.reference = wrong
    assert not helper.check_eager(torch_proxy, case, (x,), x*2)['eligible']
# Exact policies retain their original complete-output checks and never rebind.
case.verification = Exact(atol=0, rtol=0)
case.reference = lambda a:(a*2,a*3)
assert not helper.check_eager(torch_proxy, case, (x,), (x*2,x*2))['eligible']
assert 'torch_npu' not in sys.modules and not hasattr(torch, 'npu')
print('CPU helper controls passed: rebind same inputs/policy; wrong scaling, extra output and bad second output rejected; NPU not initialized')
