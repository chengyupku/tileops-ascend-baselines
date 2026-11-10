"""One-shot CPU handoff; retain the existing guard queue and pair intact."""
import fcntl
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tools'))
from run_baseline_cache import identity, put

R = ROOT/'reports/baseline-cache-20261027'
D = R/'balance-half-20261028'
STOP = R/'local/STOP'
TOKEN = 'balance-half-20261028/final-ownership-handoff\n'
expected = json.loads((D/'local-half-boundary.json').read_text())['live']['identity']
plan_hash = hashlib.sha256((D/'PLAN.half.json').read_bytes()).hexdigest()
service = 'baseline-cache-local-20261027.service'
state = {'started_at':time.time(), 'identity':identity(__import__('os').getpid()),
         'previous_supervisor':expected, 'expected_plan_sha256':plan_hash,
         'device_queue_preserved':True}

def record(phase, **fields):
    state.update(at=time.time(), phase=phase, **fields)
    put(D/'LOCAL_HANDOFF.json', state)

def still_authorized():
    return (STOP.exists() and STOP.read_text()==TOKEN and not (R/'STOP').exists()
            and hashlib.sha256((R/'PLAN.json').read_bytes()).hexdigest()==plan_hash)

while True:
    if not still_authorized():
        record('held_control_or_plan_changed')
        raise SystemExit(0)
    try:
        live = identity(expected['pid']) == expected
    except FileNotFoundError:
        live = False
    if not live:
        break
    record('waiting_for_existing_pair_boundary')
    time.sleep(2)

for _ in range(60):
    pid = subprocess.check_output(['systemctl','show',service,'--property=MainPID','--value'],text=True).strip()
    if pid == '0':
        break
    if not still_authorized():
        record('held_control_changed_after_old_exit')
        raise SystemExit(0)
    time.sleep(1)
else:
    record('held_service_still_running')
    raise SystemExit(0)

live = json.loads((R/'local/LIVE.json').read_text())
if live['phase'] != 'stopped_by_control':
    record('held_not_a_normal_control_boundary', last_phase=live['phase'])
    raise SystemExit(0)
with (R/'local/supervisor.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not still_authorized():
        record('held_control_changed_at_boundary')
        raise SystemExit(0)
    STOP.unlink()
subprocess.run(['systemctl','start',service],check=True)
for _ in range(20):
    pid=int(subprocess.check_output(['systemctl','show',service,'--property=MainPID','--value'],text=True))
    if pid:
        record('resumed_new_assignment', supervisor=identity(pid))
        break
    time.sleep(.5)
else:
    record('start_did_not_produce_supervisor')
    raise SystemExit(1)
