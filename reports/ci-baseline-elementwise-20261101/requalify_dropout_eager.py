"""Correctness only; run as one foreground npu-guard command after full suite."""
import dataclasses
import json
from pathlib import Path
import sys

BASE = Path('/opt/tileops-ci/server2-20261030/overnight-20261101')
ROOT = BASE / 'ci-full-v3/workspace'
sys.path[:0] = [str(ROOT), str(ROOT/'TileOPs'), str(ROOT/'TileOPs/src'),
                str(ROOT/'docs/ascend950/dev/logroots4_20261028')]
from benchmarks.ascend.guard import require_reservation
require_reservation()
import torch
import torch_npu
from benchmarks.api import cases
from workloads.device import set_run_device
from workloads.numerics import Custom, compare_outputs
from tools.baseline_protocol import seed_case
from tools.elementwise_eager_check import check_eager
from full_digest import tensor_digest

torch.npu.set_device(0)
p = torch.npu.get_device_properties(0)
assert torch.npu.get_device_name(0) == 'Ascend950DT_9582'
assert (p.cube_core_num, p.vector_core_num) == (32, 64)
torch.npu.set_device_limit(0, cube_num=32, vector_num=64)
assert torch.npu.get_device_limit(0) == {'cube_core_num':32, 'vector_core_num':64}
set_run_device('npu')
directory = ROOT.parent/'measurements/DropoutFwdOp'
pointer = directory/'LATEST_ATTEMPT.json'
if pointer.exists():
    directory = ROOT.parent/'measurements'/json.loads(pointer.read_text())['path']
old = {r['case_id']:r for r in map(json.loads, (directory/'measurements.jsonl').read_text().splitlines())}
results = []
try:
    for case in cases('DropoutFwdOp'):
        seed_case(torch, 'DropoutFwdOp', case.id)
        inputs = case.inputs
        digest = tensor_digest(inputs)
        assert digest == old[case.id]['input_digest']
        assert old[case.id]['correct'] and old[case.id]['input_unchanged']
        reference = case.reference(*inputs)
        check = check_eager(torch, case, inputs, reference)
        assert check['eligible'] and check['verification']['checked_outputs'] == 1
        x = inputs[0].detach().cpu()
        target = reference.detach().cpu()
        actual = case.reference(*inputs).detach().cpu()
        evidence = case.workload.verification(x)
        assert isinstance(evidence, Custom) and evidence.reason == case.verification.reason
        assert evidence.controls == case.verification.controls and evidence.probe == case.verification.probe
        positive = compare_outputs(actual, target, evidence)
        assert positive.checked_outputs == positive.total_outputs == 1
        rejected = []
        for name, invalid in [('all_zero', torch.zeros_like(actual)),
                              ('wrong_scale', actual*2),
                              ('extra_output', (actual, actual))]:
            try:
                compare_outputs(invalid, target, evidence)
            except (AssertionError, ValueError, TypeError):
                rejected.append(name)
            else:
                raise AssertionError('Bad output was admitted: '+name)
        assert len(rejected) == 3 and tensor_digest(inputs) == digest
        results.append(dict(case_id=case.id, input_digest=digest, input_unchanged=True,
                            eager_correctness=check, negative_controls_rejected=rejected,
                            independent_positive=dataclasses.asdict(positive),
                            policy_reason=evidence.reason))
        case._release()
finally:
    torch.npu.synchronize()
assert len(results) == 3
Path(sys.argv[1]).write_text(json.dumps(dict(status='passed', correctness_only=True,
                    original_timings_not_modified=True, cases=results), indent=2)+'\n')
print(json.dumps(dict(status='passed', cases=len(results), negative_controls=9)))
