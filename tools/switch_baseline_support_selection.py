"""CPU-only graceful selection handoff; never signal a device process."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
host, selection = sys.argv[1:]
b = ROOT / 'reports/baseline-cache-20261027'
d = b / host
report = b / 'support-run-20261029'
token = 'support-run-20261029/' + Path(selection).name + '\n'
with (d / 'support-control.lock').open('a') as controller:
    fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (b / 'STOP').exists():
        raise RuntimeError('Campaign STOP remains authoritative')
    if (d / 'STOP').exists() and (d / 'STOP').read_text() != token:
        raise RuntimeError('Unrelated STOP remains authoritative')
    (d / 'STOP').write_text(token)
    with (d / 'supervisor.lock').open('a') as lock:
        # Current supervisor releases this only after its foreground guard exits.
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (b / 'STOP').exists() or (d / 'STOP').read_text() != token:
            raise RuntimeError('Control changed during boundary wait')
        (d / 'STOP').unlink()
    cmd = ['/usr/bin/python3', '-u', str(ROOT / 'tools/run_baseline_support_batch_v2.py'),
           '--campaign', str(b), '--host', host, '--selection', str(ROOT / selection)]
    log = (report / (host + '-expanded-supervisor.log')).open('ab')
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    identity = {'pid': proc.pid,
                'start_ticks': int(Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]),
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    result = {'at': time.time(), 'identity': identity, 'command': cmd,
              'method': 'normal pair completion then CPU supervisor lock handoff'}
    (report / (host + '-EXPANDED-LAUNCH.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
