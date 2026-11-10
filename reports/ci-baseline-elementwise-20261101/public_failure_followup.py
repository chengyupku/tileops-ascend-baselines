"""After the original suite, diagnose its known reference-sensitive cases."""
import importlib.util
import json
from pathlib import Path
import runpy
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

BASE = Path('/opt/tileops-ci/server2-20261030/overnight-20261101')
ROOT = BASE / 'ci-full-v3/workspace'
OUT = BASE / 'public-failure-diagnosis-02'
if '--device-child' in sys.argv:
    sys.path[:0] = [str(ROOT), str(ROOT / 'TileOPs'), str(ROOT / 'TileOPs/src')]
    from benchmarks.ascend.guard import require_reservation
    require_reservation()
    for name in ('sinusoidal', 'arithmetic'):
        script = BASE / f'{name}_oracle_diagnostic.py'
        sys.argv = [str(script), str(OUT / f'{name}.json')]
        runpy.run_path(str(script), run_name='__main__')
    raise SystemExit(0)

sys.path.insert(0, str(ROOT / 'tools'))
spec = importlib.util.spec_from_file_location('controller', ROOT / 'tools/ci_elementwise32.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
from baseline_cache_guard_binding import capture_guard
original = BASE / 'public-regressions-early-01'
print(json.dumps(dict(status='waiting_for_original_suite')), flush=True)
while not (original / 'receipt.json').exists():
    time.sleep(15)
prior = json.loads((original / 'receipt.json').read_text())
failures = [dict(classname=c.get('classname'), name=c.get('name'))
            for c in ET.parse(original / 'junit.xml').iter('testcase')
            if c.find('failure') is not None or c.find('error') is not None]
print(json.dumps(dict(original_status=prior['status'], failures=failures)), flush=True)
frozen = controller.fingerprint()
assert frozen == prior['sources'], 'Original suite source changed'
OUT.mkdir(exist_ok=False)
controller.write(BASE / 'before-public-source-fix.json', frozen)
before = capture_guard()
owner = 'ci/elementwise32/public-oracle-diagnosis/' + str(time.time_ns())
command = ['npu-guard', 'run', '--owner', owner, '--wait', '600', '--timeout', '1800', '--',
           sys.executable, str(Path(__file__).resolve()), '--device-child']
started = time.time()
with (OUT / 'run.log').open('x') as log:
    result = subprocess.run(command, cwd=ROOT / 'TileOPs', stdout=log, stderr=subprocess.STDOUT)
guard = json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
assert guard['owner'] == owner
controller.write(OUT / 'guard.json', guard)
assert guard['phase'] == 'finished' and guard['performance_valid'] is True
assert not guard.get('conflict_detected') and not guard.get('visibility_lost')
assert before['files'] == capture_guard()['files'] and controller.fingerprint() == frozen
all_cpu = False
if result.returncode == 0 and guard['workload_exit_code'] == 0:
    sinusoidal = json.loads((OUT / 'sinusoidal.json').read_text())
    arithmetic = json.loads((OUT / 'arithmetic.json').read_text())
    all_cpu = sinusoidal['status'] == 'passed' and arithmetic['all_cpu_bitwise']
receipt = dict(status='passed' if all_cpu else 'needs_investigation', exit_code=result.returncode,
               original_suite_failures=failures, sources=frozen, command=command,
               started_at=started, finished_at=time.time(), guard_implementation=before['files'],
               independent_cpu_checks_passed=all_cpu,
               original_suite_receipt_sha256=controller.sha(original / 'receipt.json'),
               scripts={name:controller.sha(BASE / name) for name in
                        ('public_failure_followup.py', 'sinusoidal_oracle_diagnostic.py',
                         'arithmetic_oracle_diagnostic.py')})
controller.write(OUT / 'receipt.json', receipt)
print(json.dumps(dict(status=receipt['status'], independent_cpu_checks_passed=all_cpu,
                     exit_code=result.returncode, path=str(OUT))), flush=True)
raise SystemExit(0 if all_cpu else 2)
