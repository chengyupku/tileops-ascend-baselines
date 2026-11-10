"""Wait for the frozen batch, then run guarded independent regressions and export."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

BASE = Path('/opt/tileops-ci/server2-20261030/overnight-20261101')
JOB = BASE / 'ci-full-v3'
ROOT = JOB / 'workspace'
sys.path.insert(0, str(ROOT / 'tools'))
spec = importlib.util.spec_from_file_location('controller', ROOT / 'tools/ci_elementwise32.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
from baseline_cache_guard_binding import capture_guard

out = BASE / 'public-regressions-corrected-full-01'
out.mkdir(exist_ok=False)
frozen = controller.fingerprint()
before = capture_guard()
owner = 'ci/elementwise32/public/' + str(time.time_ns())
command = ['npu-guard', 'run', '--owner', owner, '--wait', '600', '--timeout', '5400', '--',
           sys.executable, str(ROOT / 'tools/ci_elementwise_public.py'),
           '--junitxml', str(out / 'junit.xml')]
started = time.time()
print(json.dumps(dict(phase='public_regressions', status='starting', owner=owner)), flush=True)
with (out / 'run.log').open('x') as log:
    result = subprocess.run(command, cwd=ROOT / 'TileOPs', stdout=log, stderr=subprocess.STDOUT)
errors = []
try:
    guard = json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
    assert guard['owner'] == owner
    controller.write(out / 'guard.json', guard)
    assert guard['phase'] == 'finished' and guard['performance_valid'] is True
    assert guard['workload_exit_code'] == 0
    assert not guard.get('conflict_detected') and not guard.get('visibility_lost')
    assert before['files'] == capture_guard()['files']
    assert result.returncode == 0
    assert controller.fingerprint() == frozen
except Exception as exc:
    errors.append(f'{type(exc).__name__}: {exc}')
receipt = dict(status='failed' if errors else 'passed', command=command,
               exit_code=result.returncode, errors=errors, sources=frozen,
               guard_implementation=before['files'], started_at=started, finished_at=time.time())
controller.write(out / 'receipt.json', receipt)
print(json.dumps(dict(phase='public_regressions', status=receipt['status'], errors=errors)), flush=True)
raise SystemExit(bool(errors))
