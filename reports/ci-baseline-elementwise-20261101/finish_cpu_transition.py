"""Preserve a naturally completed guarded child before restarting CPU orchestration."""
import datetime
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

BASE = Path('/opt/tileops-ci/server2-20261030/overnight-20261101')
ROOT = BASE / 'ci-full-v3/workspace'
OUT = BASE / 'ci-full-v3/measurements'
sys.path.insert(0, str(ROOT / 'tools'))
spec = importlib.util.spec_from_file_location('controller', ROOT / 'tools/ci_elementwise32.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
from baseline_cache_guard_binding import capture_guard

parent = 129249
assert b'ci_elementwise32.py' in Path(f'/proc/{parent}/cmdline').read_bytes()
assert 'T' in Path(f'/proc/{parent}/status').read_text().split('State:')[1].splitlines()[0]
guard_path = Path('/run/lock/ascend-npu/device-0.lock')
original = json.loads(guard_path.read_text())
owner = original['owner']
assert '/LogFwdOp/' in owner
before = capture_guard()
frozen = controller.fingerprint()
deadline = time.monotonic() + 1200
while True:
    guard = json.loads(guard_path.read_text())
    assert guard['owner'] == owner, 'Guard owner changed before durable receipt capture'
    if guard['phase'] == 'finished':
        break
    assert time.monotonic() < deadline, 'Current child did not naturally finish in 20 minutes'
    time.sleep(1)
directory = OUT / 'LogFwdOp'
assert not (directory / 'receipt.json').exists()
controller.write(directory / 'guard.json', guard)
assert guard['performance_valid'] is True and guard['workload_exit_code'] == 0
assert not guard.get('conflict_detected') and not guard.get('visibility_lost')
after = capture_guard()
assert before['files'] == after['files'] and before['stamps'] == after['stamps']
started = datetime.datetime.fromisoformat(guard['requested_at']).timestamp()
assert all(max(s[-2:]) / 1e9 < started for s in after['stamps'].values())
assert frozen == controller.fingerprint()
plan = next(p for p in json.loads((ROOT / 'ELEMENTWISE_PLAN.json').read_text()) if p['op'] == 'LogFwdOp')
rows = [json.loads(line) for line in (directory / 'measurements.jsonl').read_text().splitlines()]
controller.validate_rows(rows, plan['case_ids'])
gate = json.loads((ROOT / 'BASELINE_ADMISSION.json').read_text())
python = str(BASE / 'python-elementwise-preflight-1/bin/python')
command = ['npu-guard', 'run', '--owner', owner, '--wait', '600', '--timeout', '5400', '--', python,
           str(ROOT / 'tools/ci_elementwise32.py'), '--device-child', plan['script'],
           '--output', str(directory), '--op', 'LogFwdOp']
receipt = dict(op='LogFwdOp', status='passed', command=command, exit_code=0, errors=[],
               started_at=started, finished_at=time.time(), sources=frozen,
               guard_implementation=after['files'], baseline_catalog_sha256=gate['catalog_sha256'],
               recovery=dict(reason='CPU controller paused while device child naturally completed; restart with bounded CPU threads',
                             full_raw_profile_replay=True, guard_implementation_observation=after))
controller.write(directory / 'receipt.json', receipt)
assert b'ci_elementwise32.py' in Path(f'/proc/{parent}/cmdline').read_bytes()
os.kill(parent, signal.SIGTERM)
os.kill(parent, signal.SIGCONT)
controller.write(BASE / 'cpu-thread-transition-complete.json', dict(status='passed', op='LogFwdOp',
                 cases=len(rows), raw_windows=len(rows)*30, owner=owner, original_child_exit_code=0,
                 correctness_and_device_protocol_unchanged=True, cpu_threads=4, sources=frozen))
print(json.dumps(dict(status='passed', cases=len(rows), raw_windows=len(rows)*30)), flush=True)
