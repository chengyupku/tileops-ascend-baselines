"""Finish only source-dependent checks after the already queued Log1p M4."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(HERE.parents[3])
while True:
    paths = list((HERE / 'requests').glob('final-Log1pFwdOp-v6_*/status.json'))
    if paths:
        receipt = max((json.loads(p.read_text()) for p in paths), key=lambda r: r['completed_at'])
        assert receipt['guard_exit_code'] == 0, receipt
        break
    time.sleep(5)
# The old CPU coordinator was paused before it could repeat unchanged opponents.
# Its guarded child has finished, so terminating only that parent is now safe.
os.kill(1390006, signal.SIGTERM)
os.kill(1390006, signal.SIGCONT)
jobs = [
    ('edges-Log1pFwdOp-v6', 'edges.py', ['--op', 'Log1pFwdOp', '--output', str(REL / 'edges/edges-Log1pFwdOp-v6.json')], 600),
    ('pytest-Log1pFwdOp-v6', 'invoke_pytest.py', ['--kernel-module', 'log1p', 'TileOPs/tests/ops/test_unary_math.py::test_log1p', 'TileOPs/tests/ops/test_unary_math.py::test_log1p_edge', '-q', '--junitxml=' + str(REL / 'edges/pytest-Log1pFwdOp-v6.xml')], 360),
    ('profile-Log1pFwdOp-v6', 'diagnose.py', ['--op', 'Log1pFwdOp', '--output', str(REL / 'profiles/Log1pFwdOp-v6')], 240),
]
for label, script, args, timeout in jobs:
    subprocess.run(['python3', str(HERE / 'remote.py'), '--label', label,
                    '--timeout', str(timeout), '--', str(REL / script), *args], check=True)
subprocess.run(['python3', str(HERE / 'fetch.py'), 'final', 'edges', 'profiles'], check=True)
(HERE / 'log1p-v6-status.json').write_text(json.dumps({'stage': 'complete'}))
