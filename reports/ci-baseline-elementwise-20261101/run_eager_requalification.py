"""CPU supervisor: guarded correctness-only requalification, durable supplement."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

BASE = Path('/opt/tileops-ci/server2-20261030/overnight-20261101')
ROOT = BASE/'ci-full-v3/workspace'
sys.path.insert(0, str(ROOT/'tools'))
spec = importlib.util.spec_from_file_location('controller', ROOT/'tools/ci_elementwise32.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
from baseline_cache_guard_binding import capture_guard

summary = json.loads((ROOT.parent/'measurements/STATE.json').read_text())
assert summary['status'] == 'passed' and summary['completed'] == summary['expected'] == 70
output = BASE/'eager-requalification-01'
output.mkdir(exist_ok=False)
before = controller.fingerprint()
guard_impl = capture_guard()['files']
owner = 'ci/elementwise32/dropout-eager-requalification/'+str(time.time_ns())
command = ['npu-guard','run','--owner',owner,'--wait','600','--timeout','600','--',
           sys.executable,str(BASE/'requalify_dropout_eager.py'),str(output/'validation.json')]
with (output/'run.log').open('x') as log:
    run = subprocess.run(command, cwd=ROOT/'TileOPs', stdout=log, stderr=subprocess.STDOUT)
guard = json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
controller.write(output/'guard.json', guard)
assert run.returncode == 0 and guard['owner'] == owner
assert guard['phase'] == 'finished' and guard['performance_valid'] is True
assert guard['workload_exit_code'] == 0 and not guard.get('conflict_detected') and not guard.get('visibility_lost')
assert capture_guard()['files'] == guard_impl and controller.fingerprint() == before
validation = json.loads((output/'validation.json').read_text())
assert validation['status'] == 'passed' and validation['correctness_only'] is True
assert validation['original_timings_not_modified'] is True and len(validation['cases']) == 3
directory = ROOT.parent/'measurements/DropoutFwdOp'
pointer = directory/'LATEST_ATTEMPT.json'
if pointer.exists():
    directory = ROOT.parent/'measurements'/json.loads(pointer.read_text())['path']
rows = {r['case_id']:r for r in map(json.loads, (directory/'measurements.jsonl').read_text().splitlines())}
assert set(rows) == {r['case_id'] for r in validation['cases']}
for case in validation['cases']:
    assert case['input_digest'] == rows[case['case_id']]['input_digest']
    assert case['input_unchanged'] and case['eager_correctness']['eligible']
    v = case['eager_correctness']['verification']
    assert v['checked_outputs'] == v['total_outputs'] == 1
    assert case['negative_controls_rejected'] == ['all_zero','wrong_scale','extra_output']
receipt = dict(status='passed', sources=before, command=command, guard_implementation=guard_impl,
               original_measurements_sha256=controller.sha(directory/'measurements.jsonl'),
               original_receipt_sha256=controller.sha(directory/'receipt.json'),
               validation_path=str(output/'validation.json'), validation_sha256=controller.sha(output/'validation.json'),
               guard_path=str(output/'guard.json'), guard_sha256=controller.sha(output/'guard.json'),
               correction='Original Custom validator captured NPU input while outputs were moved to CPU; unchanged policy rebound to identical CPU input values. Timing records remain immutable.',
               production_kernel_changes=[], timing_changes=[], cases=validation['cases'])
controller.write(output/'receipt.json', receipt)
assert not (directory/'EAGER_READMISSION.json').exists()
controller.write(directory/'EAGER_READMISSION.json', receipt)
print(json.dumps(dict(status='passed', cases=3, negative_controls_rejected=9,
                     original_measurements_sha256=receipt['original_measurements_sha256'])))
